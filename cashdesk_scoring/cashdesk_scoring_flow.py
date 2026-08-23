import os
from datetime import date
from typing import Any, Dict, Tuple

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.blocks.system import Secret
from toolbox import oracle

from cashdesk_forecast import (
    load_settings,
    load_source_data,
    score_from_loaded_data,
)
from oracle_writer import delete_score_date_rows, insert_predictions


SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")
LOGIN_CDW = "sb_analytics"
PASSWORD_CDW = Secret.load("pass-sb-analytics").get()
ENGINE_CDW = oracle.create_engine_cdw(LOGIN_CDW, PASSWORD_CDW)

TASK_KWARGS = {
    "retries": 0,
    "tags": ["scoring"],
}

FLOW_KWARGS = {
    "retries": 1,
    "retry_delay_seconds": 60,
}


@task(name="reading_data", **TASK_KWARGS)
def read_scoring_data(
    settings_path: str,
    score_date: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    logger = get_run_logger()
    settings = load_settings(settings_path)
    loaded = load_source_data(ENGINE_CDW, settings, score_date)
    logger.info(
        "Чтение данных: score_date=%s, касс=%s, строк фактов=%s, "
        "строк истории preds=%s",
        loaded["score_date"].date(),
        len(loaded["universe_df"]),
        len(loaded["daily_df"]),
        len(loaded["preds_history_df"]),
    )
    return loaded, settings


@task(name="scoring", **TASK_KWARGS)
def score_cashdesks(
    loaded: Dict[str, Any],
    settings: Dict[str, Any],
) -> pd.DataFrame:
    logger = get_run_logger()
    preds_df = score_from_loaded_data(settings, loaded)
    logger.info(
        "Скоринг: строк=%s, касс=%s, закрытых=%s, квантиль=%s",
        len(preds_df),
        preds_df["atdtmco_cashdesk_code"].nunique()
        if not preds_df.empty
        else 0,
        int(preds_df["is_closed"].sum()) if not preds_df.empty else 0,
        float(preds_df["quantile_used"].iloc[0])
        if not preds_df.empty
        else None,
    )
    return preds_df


@task(name="delete_score_date", **TASK_KWARGS)
def delete_predictions(
    settings_path: str,
    score_date: str,
) -> int:
    logger = get_run_logger()
    settings = load_settings(settings_path)
    deleted = delete_score_date_rows(
        ENGINE_CDW,
        settings["preds_table"],
        score_date,
    )
    logger.info(
        "Удалено %s строк за score_date=%s из %s",
        deleted,
        score_date,
        settings["preds_table"],
    )
    return deleted


@task(name="write_predictions", **TASK_KWARGS)
def write_predictions(
    settings_path: str,
    preds_df: pd.DataFrame,
    score_date: str,
) -> int:
    logger = get_run_logger()
    settings = load_settings(settings_path)
    runtime = settings.get("runtime", {})
    row_count = insert_predictions(
        ENGINE_CDW,
        preds_df,
        settings["preds_table"],
        score_date,
        batch_size=int(runtime.get("write_batch_size", 100_000)),
    )
    logger.info(
        "Записано %s строк в %s за score_date=%s",
        row_count,
        settings["preds_table"],
        score_date,
    )
    return row_count


@flow(name="ts_cashdesks_0826_v1", **FLOW_KWARGS)
def ts_cashdesks_0826_v1(score_date=None):
    """
    Ежедневный скоринг потребности касс.

    Parameters
    ----------
    score_date:
        Дата скоринга YYYY-MM-DD. По умолчанию — сегодня.
    """
    logger = get_run_logger()
    resolved_score_date = score_date or date.today().isoformat()
    logger.info("Ежедневный скоринг касс, score_date=%s", resolved_score_date)

    loaded, settings = read_scoring_data(SETTINGS_PATH, resolved_score_date)
    preds_df = score_cashdesks(loaded, settings)
    deleted = delete_predictions(SETTINGS_PATH, resolved_score_date)
    row_count = write_predictions(
        SETTINGS_PATH,
        preds_df,
        resolved_score_date,
    )
    logger.info(
        "Готово: удалено=%s, записано=%s в %s",
        deleted,
        row_count,
        settings["preds_table"],
    )
    return row_count


if __name__ == "__main__":
    ts_cashdesks_0826_v1()
