import json
import logging
import os
import re
import warnings
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

WEEKDAY_ALIASES = {
    "пн": 0,
    "вт": 1,
    "ср": 2,
    "чт": 3,
    "пт": 4,
    "сб": 5,
    "вс": 6,
}
WEEKDAY_RANGE_RE = re.compile(
    r"(пн|вт|ср|чт|пт|сб|вс)\s*-\s*(пн|вт|ср|чт|пт|сб|вс)",
    flags=re.IGNORECASE,
)
WEEKDAY_TOKEN_RE = re.compile(
    r"(пн|вт|ср|чт|пт|сб|вс)",
    flags=re.IGNORECASE,
)

SCHEDULE_QUERY = """
select
    e.codefem,
    e.c_name,
    e.codeibsoretail,
    rtl_dep.c_code_dblink,
    tp.wtimecorp,
    tp.wtimepriv
from ema_working_cashdesks e
left join Ods.ODS_PRX_BANKOFFICE tp
    on tp.CODEFEM = e.codefem
   and tp.dml_type_cd <> 'D'
   and tp.statetp = 'действует'
left join ods.ods_rtl_depart rtl_dep
    on rtl_dep.c_code = tp.codeibsoretail
   and rtl_dep.dml_type_cd <> 'D'
order by e.codefem
"""


def load_settings(settings_path=None):
    if settings_path is None:
        settings_path = os.path.join(
            os.path.dirname(__file__),
            "settings.json",
        )
    with open(settings_path, encoding="utf-8") as handle:
        return json.load(handle)


def _algo(settings: Dict[str, Any]) -> Dict[str, Any]:
    return settings["algorithm"]


def parse_open_weekdays(schedule_text: object) -> FrozenSet[int]:
    if schedule_text is None or (
        isinstance(schedule_text, float) and np.isnan(schedule_text)
    ):
        return frozenset()
    text_value = str(schedule_text).strip().lower()
    if not text_value or text_value in {"nan", "none"}:
        return frozenset()
    open_days: Set[int] = set()
    for segment in text_value.split(";"):
        covered = set()
        for match in WEEKDAY_RANGE_RE.finditer(segment):
            start = WEEKDAY_ALIASES[match.group(1).lower()]
            end = WEEKDAY_ALIASES[match.group(2).lower()]
            if start <= end:
                values = range(start, end + 1)
            else:
                values = list(range(start, 7)) + list(range(0, end + 1))
            open_days.update(values)
            covered.update({match.group(1).lower(), match.group(2).lower()})
        for token in WEEKDAY_TOKEN_RE.findall(segment):
            token = token.lower()
            if token not in covered:
                open_days.add(WEEKDAY_ALIASES[token])
    return frozenset(open_days)


def build_schedule_maps(
    schedule_df: pd.DataFrame,
) -> Tuple[Dict[str, FrozenSet[int]], pd.DataFrame]:
    frame = schedule_df.copy()
    frame.columns = frame.columns.str.lower()
    frame["c_code_dblink"] = frame["c_code_dblink"].astype(str).str.strip()
    frame.loc[
        frame["c_code_dblink"].isin(["", "None", "nan", "NaN"]),
        "c_code_dblink",
    ] = pd.NA

    open_weekdays: Dict[str, FrozenSet[int]] = {}
    for row in frame.itertuples(index=False):
        code = getattr(row, "c_code_dblink", None)
        if code is None or (isinstance(code, float) and np.isnan(code)):
            continue
        code = str(code).strip()
        if not code or code in {"None", "nan", "NaN"}:
            continue
        days = parse_open_weekdays(
            getattr(row, "wtimecorp", None)
        ) | parse_open_weekdays(getattr(row, "wtimepriv", None))
        if days:
            open_weekdays[code] = days
    return open_weekdays, frame


def is_cashdesk_open(
    cashdesk_code: object,
    day: pd.Timestamp,
    open_weekdays: Dict[str, FrozenSet[int]],
    holiday_dates: Set[pd.Timestamp],
) -> Optional[bool]:
    if cashdesk_code is None or (
        isinstance(cashdesk_code, float) and np.isnan(cashdesk_code)
    ):
        return None
    code = str(cashdesk_code).strip()
    days = open_weekdays.get(code)
    if days is None:
        return None
    day = pd.Timestamp(day).normalize()
    if day in holiday_dates:
        return False
    return int(day.dayofweek) in days


def is_cashdesk_closed(
    cashdesk_code: object,
    day: pd.Timestamp,
    has_source_row: bool,
    open_weekdays: Dict[str, FrozenSet[int]],
    holiday_dates: Set[pd.Timestamp],
) -> bool:
    open_status = is_cashdesk_open(
        cashdesk_code, day, open_weekdays, holiday_dates
    )
    if open_status is None:
        return not bool(has_source_row)
    return not open_status


def make_regular_series(
    cashdesk_df: pd.DataFrame,
    value_column: str,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
    exclude_date_ranges: List[List[str]],
) -> pd.Series:
    """Build daily series; missing days stay NaN (no open-day zero fill)."""
    full_index = pd.date_range(date_from, date_to, freq="D")
    observed = (
        cashdesk_df
        .drop_duplicates("calday", keep="last")
        .set_index("calday")[value_column]
        .astype(float)
    )
    series = observed.reindex(full_index)
    for exclude_start, exclude_end in exclude_date_ranges:
        excluded_mask = (
            (series.index >= pd.Timestamp(exclude_start))
            & (series.index < pd.Timestamp(exclude_end))
        )
        series.loc[excluded_mask] = np.nan
    series.index.name = "calday"
    return series


def make_future_index(y: pd.Series, steps: int) -> pd.DatetimeIndex:
    return pd.date_range(
        y.index.max() + pd.Timedelta(days=1),
        periods=steps,
        freq="D",
    )


def guarded_sarima_forecast(
    y: pd.Series,
    steps: int,
    settings: Dict[str, Any],
) -> np.ndarray:
    algo = _algo(settings)
    model = SARIMAX(
        y,
        order=tuple(algo["sarima_order"]),
        seasonal_order=tuple(algo["sarima_seasonal_order"]),
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Non-invertible starting.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Non-stationary starting.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Maximum Likelihood optimization failed.*",
        )
        fitted = model.fit(disp=False, maxiter=int(algo["sarima_maxiter"]))
    if not bool(fitted.mle_retvals.get("converged", False)):
        raise ValueError("SARIMA не сошлась")
    for root_name, roots in (("AR", fitted.arroots), ("MA", fitted.maroots)):
        root_modulus = np.abs(np.asarray(roots, dtype=complex))
        if root_modulus.size and (
            not np.isfinite(root_modulus).all()
            or (root_modulus <= 1.0).any()
        ):
            raise ValueError("Неустойчивые {}-корни".format(root_name))
    forecast = fitted.get_forecast(steps=steps).predicted_mean.to_numpy(
        dtype=float
    )
    valid_history = y.dropna().to_numpy(dtype=float)
    history_scale = max(
        1.0,
        float(np.quantile(np.abs(valid_history), 0.99)),
    )
    if not np.isfinite(forecast).all():
        raise ValueError("Нечисловой SARIMA-прогноз")
    if (
        np.abs(forecast)
        > float(algo["forecast_scale_multiplier"]) * history_scale
    ).any():
        raise ValueError("Несоразмерный SARIMA-прогноз")
    return forecast


def weekday_point_forecast(
    y: pd.Series,
    steps: int,
    clip_upper_zero: bool,
    cash_need_clip_upper: float,
) -> np.ndarray:
    y_clean = y.dropna()
    if y_clean.empty:
        raise ValueError("Нет истории для fallback")
    global_value = float(y_clean.median())
    weekday_values = y_clean.groupby(y_clean.index.dayofweek).median()
    values = np.asarray([
        weekday_values.get(day.dayofweek, global_value)
        for day in make_future_index(y, steps)
    ], dtype=float)
    if clip_upper_zero:
        values = np.minimum(values, cash_need_clip_upper)
    return values


def forecast_with_fallback(
    y: pd.Series,
    steps: int,
    clip_upper_zero: bool,
    settings: Dict[str, Any],
) -> Tuple[np.ndarray, bool]:
    algo = _algo(settings)
    try:
        if y.notna().sum() < int(algo["min_sarima_obs"]):
            raise ValueError("Недостаточно наблюдений")
        values = guarded_sarima_forecast(y, steps, settings)
        if clip_upper_zero:
            values = np.minimum(values, float(algo["cash_need_clip_upper"]))
        return values, False
    except Exception:
        try:
            return weekday_point_forecast(
                y,
                steps,
                clip_upper_zero,
                float(algo["cash_need_clip_upper"]),
            ), True
        except Exception:
            return np.zeros(steps, dtype=float), True


def weighted_empirical_quantile(
    global_values: np.ndarray,
    cashdesk_values: np.ndarray,
    quantile: float,
    shrinkage: float,
) -> float:
    global_values = np.asarray(global_values, dtype=float)
    cashdesk_values = np.asarray(cashdesk_values, dtype=float)
    global_values = global_values[np.isfinite(global_values)]
    cashdesk_values = cashdesk_values[np.isfinite(cashdesk_values)]
    if len(global_values) == 0:
        if len(cashdesk_values) == 0:
            return 0.0
        return float(np.quantile(cashdesk_values, quantile))
    if len(cashdesk_values) == 0:
        return float(np.quantile(global_values, quantile))
    cashdesk_weight = len(cashdesk_values) / (
        len(cashdesk_values) + shrinkage
    )
    values = np.concatenate([global_values, cashdesk_values])
    weights = np.concatenate([
        np.full(
            len(global_values),
            (1.0 - cashdesk_weight) / len(global_values),
        ),
        np.full(
            len(cashdesk_values),
            cashdesk_weight / len(cashdesk_values),
        ),
    ])
    order = np.argsort(values)
    cumulative_weights = np.cumsum(weights[order])
    index = np.searchsorted(
        cumulative_weights,
        quantile * cumulative_weights[-1],
        side="left",
    )
    return float(values[order][min(index, len(values) - 1)])


def load_source_data(
    engine,
    settings: Dict[str, Any],
    score_date: pd.Timestamp,
) -> Dict[str, Any]:
    algo = _algo(settings)
    score_date = pd.Timestamp(score_date).normalize()
    report_date = score_date - pd.Timedelta(days=int(algo["data_lag_days"]))
    history_date_from = (
        report_date
        - pd.DateOffset(months=int(algo["history_months"]))
        + pd.Timedelta(days=1)
    )
    fact_date_to = score_date + pd.Timedelta(
        days=int(algo["forecast_days"])
    )
    error_lookback_from = report_date - pd.Timedelta(
        days=int(algo["error_window_days"]) + int(algo["forecast_days"])
    )

    source_query = """
    select
        atdtmco_cashdesk_name,
        atdtmco_cashdesk_name_trn,
        atdtmco_cashdesk_code,
        atdtmco_calday,
        atdtmco_saldo_turn,
        atdtmco_ns
    from {source_table}
    where atdtmco_calday >= date '{date_from}'
      and atdtmco_calday < date '{date_to}'
    """.format(
        source_table=settings["source_table"],
        date_from=history_date_from.date().isoformat(),
        date_to=fact_date_to.date().isoformat(),
    )
    for exclude_start, exclude_end in algo.get("exclude_date_ranges", []):
        source_query += (
            "\n  and not (atdtmco_calday >= date '{}' "
            "and atdtmco_calday < date '{}')"
        ).format(exclude_start, exclude_end)

    holiday_query = """
    select cld_day_dt, hol_msk_flg
    from {calendar_table}
    where cld_day_dt >= date '{date_from}'
      and cld_day_dt < date '{date_to}'
    """.format(
        calendar_table=settings["calendar_table"],
        date_from=history_date_from.date().isoformat(),
        date_to=fact_date_to.date().isoformat(),
    )

    preds_history_query = """
    select
        score_date,
        report_date,
        atdtmco_cashdesk_code,
        atdtmco_cashdesk_name,
        forecast_date,
        atdtmco_ns_pred_central,
        atdtmco_ns_pred,
        quantile_used,
        is_closed
    from {preds_table}
    where score_date >= date '{date_from}'
      and score_date < date '{date_to}'
    """.format(
        preds_table=settings["preds_table"],
        date_from=error_lookback_from.date().isoformat(),
        date_to=score_date.date().isoformat(),
    )

    with engine.connect() as conn:
        raw_df = pd.read_sql(text(source_query), conn)
        schedule_df = pd.read_sql(text(SCHEDULE_QUERY), conn)
        holiday_df = pd.read_sql(text(holiday_query), conn)
        try:
            preds_history_df = pd.read_sql(text(preds_history_query), conn)
        except Exception as error:
            logger.warning(
                "Не удалось прочитать историю preds (%s); холодный старт",
                error,
            )
            preds_history_df = pd.DataFrame()

    raw_df.columns = raw_df.columns.str.lower()
    schedule_df.columns = schedule_df.columns.str.lower()
    holiday_df.columns = holiday_df.columns.str.lower()
    if not preds_history_df.empty:
        preds_history_df.columns = preds_history_df.columns.str.lower()

    raw_df["atdtmco_calday"] = pd.to_datetime(
        raw_df["atdtmco_calday"]
    ).dt.normalize()
    raw_df["atdtmco_cashdesk_code"] = (
        raw_df["atdtmco_cashdesk_code"].astype(str).str.strip()
    )
    holiday_df["cld_day_dt"] = pd.to_datetime(
        holiday_df["cld_day_dt"]
    ).dt.normalize()
    holiday_df["hol_msk_flg"] = pd.to_numeric(
        holiday_df["hol_msk_flg"], errors="coerce"
    ).fillna(0).astype(int)

    daily_df = (
        raw_df
        .sort_values(["atdtmco_cashdesk_name", "atdtmco_calday"])
        .groupby(
            ["atdtmco_cashdesk_name", "atdtmco_calday"],
            as_index=False,
            dropna=False,
        )
        .agg(
            atdtmco_cashdesk_name_trn=("atdtmco_cashdesk_name_trn", "last"),
            atdtmco_cashdesk_code=("atdtmco_cashdesk_code", "last"),
            atdtmco_saldo_turn_fact=(
                "atdtmco_saldo_turn",
                lambda values: values.sum(min_count=1),
            ),
            atdtmco_ns_daily_min_raw=("atdtmco_ns", "min"),
        )
        .rename(columns={"atdtmco_calday": "calday"})
        .sort_values(["atdtmco_cashdesk_name", "calday"])
        .reset_index(drop=True)
    )
    daily_df["atdtmco_ns_fact"] = pd.to_numeric(
        daily_df["atdtmco_ns_daily_min_raw"], errors="coerce"
    ).clip(upper=float(algo["cash_need_clip_upper"]))

    open_weekdays, schedule_clean = build_schedule_maps(schedule_df)
    holiday_dates = set(
        holiday_df.loc[holiday_df["hol_msk_flg"].eq(1), "cld_day_dt"].tolist()
    )

    code_to_names = (
        daily_df
        .dropna(subset=["atdtmco_cashdesk_code"])
        .sort_values("calday")
        .groupby("atdtmco_cashdesk_code", as_index=False)
        .agg(
            atdtmco_cashdesk_name=("atdtmco_cashdesk_name", "last"),
            atdtmco_cashdesk_name_trn=("atdtmco_cashdesk_name_trn", "last"),
        )
    )
    universe = schedule_clean.dropna(subset=["c_code_dblink"])[
        ["c_code_dblink", "c_name"]
    ].drop_duplicates("c_code_dblink").rename(
        columns={
            "c_code_dblink": "atdtmco_cashdesk_code",
            "c_name": "directory_name",
        }
    )
    universe = universe.merge(
        code_to_names,
        on="atdtmco_cashdesk_code",
        how="left",
    )
    universe["atdtmco_cashdesk_name"] = universe[
        "atdtmco_cashdesk_name"
    ].combine_first(universe["directory_name"])
    universe["atdtmco_cashdesk_name_trn"] = universe[
        "atdtmco_cashdesk_name_trn"
    ].combine_first(universe["directory_name"])

    if not preds_history_df.empty:
        for column in ("score_date", "report_date", "forecast_date"):
            preds_history_df[column] = pd.to_datetime(
                preds_history_df[column]
            ).dt.normalize()
        preds_history_df["atdtmco_cashdesk_code"] = (
            preds_history_df["atdtmco_cashdesk_code"].astype(str).str.strip()
        )
        preds_history_df["forecast_step"] = (
            preds_history_df["forecast_date"]
            - preds_history_df["score_date"]
        ).dt.days.astype(int)

    return {
        "score_date": score_date,
        "report_date": report_date,
        "history_date_from": history_date_from,
        "daily_df": daily_df,
        "universe_df": universe,
        "open_weekdays": open_weekdays,
        "holiday_dates": holiday_dates,
        "preds_history_df": preds_history_df,
    }


def resolve_quantile(
    settings: Dict[str, Any],
    score_date: pd.Timestamp,
    report_date: pd.Timestamp,
    preds_history_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> float:
    algo = _algo(settings)
    initial_q = float(algo["initial_quantile"])
    if preds_history_df.empty:
        return initial_q

    previous = preds_history_df[
        preds_history_df["score_date"] < score_date
    ]
    if previous.empty:
        quantile = initial_q
    else:
        last_score = previous["score_date"].max()
        quantile = float(
            previous.loc[
                previous["score_date"].eq(last_score),
                "quantile_used",
            ].iloc[0]
        )

    first_day = preds_history_df[
        preds_history_df["score_date"].eq(report_date)
        & preds_history_df["forecast_date"].eq(report_date)
        & (~preds_history_df["is_closed"].astype(bool))
    ].copy()
    if first_day.empty:
        return quantile

    facts = daily_df[[
        "atdtmco_cashdesk_code",
        "calday",
        "atdtmco_ns_fact",
    ]].rename(columns={"calday": "forecast_date"})
    first_day = first_day.merge(
        facts,
        on=["atdtmco_cashdesk_code", "forecast_date"],
        how="left",
    )
    update_df = first_day[first_day["atdtmco_ns_fact"] < 0]
    if len(update_df) < int(algo["adaptive_min_update_rows"]):
        return quantile

    breach = float(
        (update_df["atdtmco_ns_fact"] < update_df["atdtmco_ns_pred"]).mean()
    )
    quantile = float(np.clip(
        quantile
        + float(algo["adaptive_gamma"])
        * (float(algo["target_breach_rate"]) - breach),
        float(algo["adaptive_quantile_min"]),
        float(algo["adaptive_quantile_max"]),
    ))
    logger.info(
        "Adaptive q update on %s: breach=%.3f -> q=%.3f (n=%s)",
        report_date.date(),
        breach,
        quantile,
        len(update_df),
    )
    return quantile


def compute_scale(
    daily_df: pd.DataFrame,
    cashdesk_code: object,
    report_date: pd.Timestamp,
    window_days: int,
    min_scale_rows: int,
    global_scale: float,
) -> float:
    report_date = pd.Timestamp(report_date).normalize()
    date_from = report_date - pd.Timedelta(days=window_days - 1)
    values = daily_df.loc[
        (daily_df["atdtmco_cashdesk_code"].astype(str) == str(cashdesk_code))
        & (daily_df["calday"] >= date_from)
        & (daily_df["calday"] <= report_date)
        & (daily_df["atdtmco_ns_fact"] < 0),
        "atdtmco_ns_fact",
    ].abs()
    if len(values) >= min_scale_rows:
        scale = float(values.median())
        if np.isfinite(scale) and scale > 0:
            return scale
    return float(global_scale)


def compute_global_scale(
    daily_df: pd.DataFrame,
    report_date: pd.Timestamp,
    window_days: int,
) -> float:
    report_date = pd.Timestamp(report_date).normalize()
    date_from = report_date - pd.Timedelta(days=window_days - 1)
    values = daily_df.loc[
        (daily_df["calday"] >= date_from)
        & (daily_df["calday"] <= report_date)
        & (daily_df["atdtmco_ns_fact"] < 0),
        "atdtmco_ns_fact",
    ].abs()
    scale = float(values.median()) if len(values) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    return scale


def build_error_pool(
    settings: Dict[str, Any],
    report_date: pd.Timestamp,
    preds_history_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    algo = _algo(settings)
    if preds_history_df.empty:
        return pd.DataFrame(columns=[
            "atdtmco_cashdesk_code",
            "forecast_date",
            "forecast_step",
            "normalized_error",
        ])

    window = int(algo["error_window_days"])
    min_scale_rows = int(algo["min_scale_rows"])
    error_date_from = report_date - pd.Timedelta(days=window - 1)
    hist = preds_history_df[
        (preds_history_df["forecast_date"] >= error_date_from)
        & (preds_history_df["forecast_date"] <= report_date)
        & (~preds_history_df["is_closed"].astype(bool))
    ].copy()
    facts = daily_df[[
        "atdtmco_cashdesk_code",
        "calday",
        "atdtmco_ns_fact",
    ]].rename(columns={"calday": "forecast_date"})
    hist = hist.merge(
        facts,
        on=["atdtmco_cashdesk_code", "forecast_date"],
        how="inner",
    )
    hist = hist[
        (hist["atdtmco_ns_fact"] < 0)
        & hist["atdtmco_ns_pred_central"].notna()
    ].copy()
    if hist.empty:
        return pd.DataFrame(columns=[
            "atdtmco_cashdesk_code",
            "forecast_date",
            "forecast_step",
            "normalized_error",
        ])

    # scale на момент исторического скоринга восстанавливаем из фактов
    scale_keys = hist[[
        "atdtmco_cashdesk_code",
        "report_date",
    ]].drop_duplicates()
    global_by_report = {
        pd.Timestamp(day).normalize(): compute_global_scale(
            daily_df, day, window
        )
        for day in scale_keys["report_date"].unique()
    }
    scale_rows = []
    for row in scale_keys.itertuples(index=False):
        report = pd.Timestamp(row.report_date).normalize()
        scale_rows.append({
            "atdtmco_cashdesk_code": row.atdtmco_cashdesk_code,
            "report_date": report,
            "scale": compute_scale(
                daily_df,
                row.atdtmco_cashdesk_code,
                report,
                window,
                min_scale_rows,
                global_by_report[report],
            ),
        })
    scale_df = pd.DataFrame(scale_rows)
    hist = hist.merge(
        scale_df,
        on=["atdtmco_cashdesk_code", "report_date"],
        how="left",
        validate="many_to_one",
    )
    hist = hist[hist["scale"].notna() & (hist["scale"] > 0)].copy()
    hist["normalized_error"] = (
        hist["atdtmco_ns_fact"] - hist["atdtmco_ns_pred_central"]
    ) / hist["scale"]
    return hist[[
        "atdtmco_cashdesk_code",
        "forecast_date",
        "forecast_step",
        "normalized_error",
    ]]


def _forecast_one_cashdesk(
    score_date: pd.Timestamp,
    report_date: pd.Timestamp,
    history_date_from: pd.Timestamp,
    cashdesk_code: str,
    cashdesk_name: object,
    cashdesk_name_trn: object,
    cashdesk_df: pd.DataFrame,
    global_scale: float,
    settings: Dict[str, Any],
) -> pd.DataFrame:
    algo = _algo(settings)
    model_steps = int(algo["forecast_days"]) + int(algo["data_lag_days"]) - 1
    window = int(algo["error_window_days"])

    if cashdesk_df.empty:
        ns_series = pd.Series(
            dtype=float,
            index=pd.date_range(history_date_from, report_date, freq="D"),
        )
        ns_series.index.name = "calday"
        saldo_series = ns_series.copy()
    else:
        ns_series = make_regular_series(
            cashdesk_df,
            "atdtmco_ns_fact",
            history_date_from,
            report_date,
            algo.get("exclude_date_ranges", []),
        )
        saldo_series = make_regular_series(
            cashdesk_df,
            "atdtmco_saldo_turn_fact",
            history_date_from,
            report_date,
            algo.get("exclude_date_ranges", []),
        )

    ns_values, _fallback_ns = forecast_with_fallback(
        ns_series, model_steps, True, settings
    )
    saldo_values, _fallback_saldo = forecast_with_fallback(
        saldo_series, model_steps, False, settings
    )

    date_from = report_date - pd.Timedelta(days=window - 1)
    local_values = ns_series[
        (ns_series.index >= date_from) & (ns_series < 0)
    ].abs().dropna()
    scale = (
        float(local_values.median())
        if len(local_values) >= int(algo["min_scale_rows"])
        else global_scale
    )

    future_index = make_future_index(ns_series, model_steps)
    result_df = pd.DataFrame({
        "forecast_date": future_index,
        "atdtmco_ns_pred_central": ns_values,
        "atdtmco_saldo_turn_pred": saldo_values,
    })
    result_df = result_df[
        (result_df["forecast_date"] >= score_date)
        & (
            result_df["forecast_date"]
            < score_date + pd.Timedelta(days=int(algo["forecast_days"]))
        )
    ].copy()
    result_df.insert(0, "score_date", score_date)
    result_df.insert(1, "report_date", report_date)
    result_df.insert(2, "atdtmco_cashdesk_code", cashdesk_code)
    result_df.insert(3, "atdtmco_cashdesk_name", cashdesk_name)
    result_df.insert(4, "atdtmco_cashdesk_name_trn", cashdesk_name_trn)
    result_df["scale"] = scale
    result_df["forecast_step"] = (
        result_df["forecast_date"] - result_df["score_date"]
    ).dt.days.astype(int)
    return result_df


def run_daily_score(
    engine,
    settings: Dict[str, Any],
    score_date,
) -> pd.DataFrame:
    algo = _algo(settings)
    loaded = load_source_data(engine, settings, score_date)
    score_date = loaded["score_date"]
    report_date = loaded["report_date"]
    history_date_from = loaded["history_date_from"]
    daily_df = loaded["daily_df"]
    universe_df = loaded["universe_df"]
    open_weekdays = loaded["open_weekdays"]
    holiday_dates = loaded["holiday_dates"]
    preds_history_df = loaded["preds_history_df"]

    if universe_df.empty:
        raise RuntimeError("Справочник ema_working_cashdesks пуст")

    quantile = resolve_quantile(
        settings,
        score_date,
        report_date,
        preds_history_df,
        daily_df,
    )
    error_pool_df = build_error_pool(
        settings, report_date, preds_history_df, daily_df
    )

    window = int(algo["error_window_days"])
    scale_date_from = report_date - pd.Timedelta(days=window - 1)
    global_values = daily_df.loc[
        (daily_df["calday"] >= scale_date_from)
        & (daily_df["calday"] <= report_date)
        & (daily_df["atdtmco_ns_fact"] < 0),
        "atdtmco_ns_fact",
    ].abs()
    global_scale = float(global_values.median()) if len(global_values) else 1.0
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    available_df = daily_df[
        (daily_df["calday"] >= history_date_from)
        & (daily_df["calday"] <= report_date)
    ]
    tasks = []
    for row in universe_df.itertuples(index=False):
        code = str(row.atdtmco_cashdesk_code)
        cashdesk_df = available_df[
            available_df["atdtmco_cashdesk_code"].astype(str).eq(code)
        ].copy()
        if cashdesk_df.empty:
            # fallback: match by name if code missing in facts
            cashdesk_df = available_df[
                available_df["atdtmco_cashdesk_name"].eq(
                    row.atdtmco_cashdesk_name
                )
            ].copy()
        tasks.append((
            code,
            row.atdtmco_cashdesk_name,
            row.atdtmco_cashdesk_name_trn,
            cashdesk_df,
        ))

    logger.info(
        "Scoring %s: cashdesks=%s, q=%.3f, error_rows=%s",
        score_date.date(),
        len(tasks),
        quantile,
        len(error_pool_df),
    )

    central_parts = [
        _forecast_one_cashdesk(
            score_date,
            report_date,
            history_date_from,
            cashdesk_code,
            cashdesk_name,
            cashdesk_name_trn,
            cashdesk_df,
            global_scale,
            settings,
        )
        for cashdesk_code, cashdesk_name, cashdesk_name_trn, cashdesk_df
        in tasks
    ]
    central_df = pd.concat(central_parts, ignore_index=True)

    fact_lookup = daily_df[[
        "atdtmco_cashdesk_code",
        "calday",
        "atdtmco_ns_fact",
        "atdtmco_saldo_turn_fact",
    ]].copy()
    fact_lookup["atdtmco_cashdesk_code"] = (
        fact_lookup["atdtmco_cashdesk_code"].astype(str)
    )
    fact_lookup = fact_lookup.rename(columns={"calday": "forecast_date"})
    fact_lookup["has_source_row"] = True

    central_df = central_df.merge(
        fact_lookup,
        on=["atdtmco_cashdesk_code", "forecast_date"],
        how="left",
        validate="many_to_one",
    )
    has_source = central_df["has_source_row"].fillna(False).astype(bool)
    central_df["is_closed"] = [
        is_cashdesk_closed(
            code, day, has_row, open_weekdays, holiday_dates
        )
        for code, day, has_row in zip(
            central_df["atdtmco_cashdesk_code"],
            central_df["forecast_date"],
            has_source,
        )
    ]

    all_errors = error_pool_df["normalized_error"].to_numpy(dtype=float)
    correction_rows = []
    for forecast_step, step_df in central_df.groupby("forecast_step", sort=True):
        step_errors = error_pool_df[
            error_pool_df["forecast_step"].eq(forecast_step)
        ]
        step_global = step_errors["normalized_error"].to_numpy(dtype=float)
        if len(step_global) == 0:
            step_global = all_errors
        by_code = {
            code: group["normalized_error"].to_numpy(dtype=float)
            for code, group in step_errors.groupby("atdtmco_cashdesk_code")
        }
        for code in step_df["atdtmco_cashdesk_code"].unique():
            correction_rows.append({
                "atdtmco_cashdesk_code": code,
                "forecast_step": forecast_step,
                "hierarchical_correction": weighted_empirical_quantile(
                    step_global,
                    by_code.get(code, np.array([], dtype=float)),
                    quantile,
                    float(algo["shrinkage"]),
                ),
            })
    correction_df = pd.DataFrame(correction_rows)
    if correction_df.empty:
        central_df["hierarchical_correction"] = 0.0
    else:
        central_df = central_df.merge(
            correction_df,
            on=["atdtmco_cashdesk_code", "forecast_step"],
            how="left",
            validate="many_to_one",
        )
        central_df["hierarchical_correction"] = central_df[
            "hierarchical_correction"
        ].fillna(0.0)

    central_df["quantile_used"] = quantile
    central_df["atdtmco_ns_pred"] = np.minimum(
        central_df["atdtmco_ns_pred_central"]
        + central_df["hierarchical_correction"] * central_df["scale"],
        float(algo["cash_need_clip_upper"]),
    )
    closed_mask = central_df["is_closed"].astype(bool)
    central_df.loc[closed_mask, "atdtmco_ns_pred"] = 0.0
    central_df.loc[closed_mask, "atdtmco_saldo_turn_pred"] = 0.0

    result_df = central_df[[
        "score_date",
        "report_date",
        "atdtmco_cashdesk_code",
        "atdtmco_cashdesk_name",
        "atdtmco_cashdesk_name_trn",
        "forecast_date",
        "atdtmco_ns_pred_central",
        "atdtmco_ns_pred",
        "atdtmco_saldo_turn_pred",
        "quantile_used",
        "is_closed",
    ]].copy()
    result_df["is_closed"] = result_df["is_closed"].astype(int)

    numeric_columns = [
        "atdtmco_ns_pred_central",
        "atdtmco_ns_pred",
        "atdtmco_saldo_turn_pred",
        "quantile_used",
    ]
    if not np.isfinite(result_df[numeric_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("В итоговом прогнозе есть нечисловые значения")
    return result_df
