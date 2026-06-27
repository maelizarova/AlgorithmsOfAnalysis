from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd


RESULT_COLUMNS = [
    "claim_num",
    "created",
    "score_date",
    "product",
    "theme",
    "category",
    "classifier_name",
    "type",
    "class",
    "sub_class",
    "eval",
]


def compute_score_window(score_date: date | datetime | str | None = None, lookback_days: int = 1) -> tuple[str, str]:
    if lookback_days < 1:
        raise ValueError("lookback_days must be greater than zero")

    resolved_score_date = _coerce_date(score_date)
    start_date = resolved_score_date - timedelta(days=lookback_days)
    return start_date.isoformat(), resolved_score_date.isoformat()


def filter_source_data(
    data: pd.DataFrame,
    *,
    product: str | None = None,
    theme: str | None = None,
    category: str | None = None,
) -> pd.DataFrame:
    result = data.copy()
    for column, value in (("product", product), ("theme", theme), ("category", category)):
        if value:
            result = result[result[column] == value]
    return result.reset_index(drop=True)


def build_result_frame(
    classified_df: pd.DataFrame,
    judge_issues_df: pd.DataFrame,
    judge_requests_df: pd.DataFrame,
    *,
    score_date: date | datetime | str,
    classifier_name: str,
) -> pd.DataFrame:
    score_timestamp = pd.to_datetime(_coerce_date(score_date))
    rows = []
    rows.extend(
        _iter_result_rows(
            classified_df,
            judge_issues_df,
            labels_column="issues_pred",
            label_type="issue",
            score_date=score_timestamp,
            classifier_name=classifier_name,
        )
    )
    rows.extend(
        _iter_result_rows(
            classified_df,
            judge_requests_df,
            labels_column="requested_actions_pred",
            label_type="req_action",
            score_date=score_timestamp,
            classifier_name=classifier_name,
        )
    )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _iter_result_rows(
    classified_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    *,
    labels_column: str,
    label_type: str,
    score_date: pd.Timestamp,
    classifier_name: str,
) -> list[dict[str, Any]]:
    rows = []
    judge_rows = _group_judge_rows(judge_df)

    for row_id, row in classified_df.reset_index(drop=True).iterrows():
        labels = _labels_to_list(row.get(labels_column))
        verdicts = judge_rows.get(row_id, [])
        for label_idx, label in enumerate(labels):
            if not isinstance(label, dict):
                continue
            verdict = verdicts[label_idx] if label_idx < len(verdicts) else {}
            rows.append(
                {
                    "claim_num": row.get("claim_num", ""),
                    "created": row.get("created"),
                    "score_date": score_date,
                    "product": row.get("product", ""),
                    "theme": row.get("theme", ""),
                    "category": row.get("category", ""),
                    "classifier_name": classifier_name,
                    "type": label_type,
                    "class": label.get("category", ""),
                    "sub_class": label.get("sub_category", ""),
                    "eval": _is_eval_ok(verdict),
                }
            )
    return rows


def _group_judge_rows(judge_df: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    if judge_df.empty or "row_id" not in judge_df.columns:
        return {}

    grouped: dict[int, list[dict[str, Any]]] = {}
    for _, row in judge_df.iterrows():
        grouped.setdefault(int(row["row_id"]), []).append(row.to_dict())
    return grouped


def _is_eval_ok(verdict: dict[str, Any]) -> bool:
    return bool(verdict.get("judge_category_ok")) and bool(verdict.get("judge_sub_category_ok"))


def _labels_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return []


def _coerce_date(value: date | datetime | str | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()
