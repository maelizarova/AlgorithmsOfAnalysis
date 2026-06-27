from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, create_model


MAX_HTTP_TIMEOUT_SECONDS = 60


class ClassificationLabel(BaseModel):
    category: str
    sub_category: str
    description: str


class ClassificationAnswer(BaseModel):
    issues: list[ClassificationLabel] = Field(default_factory=list)
    requested_actions: list[ClassificationLabel] = Field(default_factory=list)


class JudgeLabelVerdict(BaseModel):
    category: str
    sub_category: str
    judge_category_ok: bool
    judge_sub_category_ok: bool
    suggested_category: str = ""
    suggested_sub_category: str = ""


class JudgeArrayAnswer(BaseModel):
    verdicts: list[JudgeLabelVerdict]


class LLMConfig(BaseModel):
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0
    timeout_seconds: int = MAX_HTTP_TIMEOUT_SECONDS
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


def build_llm_from_config(llm_config: LLMConfig) -> ChatOpenAI:
    timeout_seconds = min(llm_config.timeout_seconds, MAX_HTTP_TIMEOUT_SECONDS)
    return ChatOpenAI(
        model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        temperature=llm_config.temperature,
        timeout=timeout_seconds,
        max_retries=0,
        extra_body=llm_config.model_kwargs or None,
    )


def build_classification_model(
    issues_taxonomy_path: str | Path,
    requests_taxonomy_path: str | Path,
) -> type[BaseModel]:
    issues = json.loads(Path(issues_taxonomy_path).read_text(encoding="utf-8"))
    requests = json.loads(Path(requests_taxonomy_path).read_text(encoding="utf-8"))

    issue_cats = sorted({item["category"] for item in issues})
    issue_subs = sorted({item["sub_category"] for item in issues})
    request_cats = sorted({item["category"] for item in requests})
    request_subs = sorted({item["sub_category"] for item in requests})

    IssueCatEnum = _make_str_enum("IssueCatEnum", issue_cats)
    IssueSubEnum = _make_str_enum("IssueSubEnum", issue_subs)
    RequestCatEnum = _make_str_enum("RequestCatEnum", request_cats)
    RequestSubEnum = _make_str_enum("RequestSubEnum", request_subs)

    IssueLabel = create_model(
        "IssueLabel",
        category=(IssueCatEnum, ...),
        sub_category=(IssueSubEnum, ...),
        description=(str, ...),
    )
    RequestLabel = create_model(
        "RequestLabel",
        category=(RequestCatEnum, ...),
        sub_category=(RequestSubEnum, ...),
        description=(str, ...),
    )
    return create_model(
        "ClassificationAnswer",
        issues=(list[IssueLabel], Field(default_factory=list)),
        requested_actions=(list[RequestLabel], Field(default_factory=list)),
    )


def build_stage1_chain(
    llm: ChatOpenAI,
    prompt_path: str | Path,
    issues_table: str,
    requests_table: str,
    product_context: str = "",
    output_model: type[BaseModel] | None = None,
) -> Any:
    prompt = ChatPromptTemplate.from_template(_read_text(prompt_path)).partial(
        issues_table=issues_table,
        requests_table=requests_table,
        product_context=product_context,
    )
    model = output_model or ClassificationAnswer
    return prompt | llm.with_structured_output(model)


def build_judge_issues_chain(
    llm: ChatOpenAI,
    prompt_path: str | Path,
    issues_table: str,
    product_context: str = "",
) -> Any:
    prompt = ChatPromptTemplate.from_template(_read_text(prompt_path)).partial(
        issues_table=issues_table,
        product_context=product_context,
    )
    return prompt | llm.with_structured_output(JudgeArrayAnswer)


def build_judge_requests_chain(
    llm: ChatOpenAI,
    prompt_path: str | Path,
    requests_table: str,
    product_context: str = "",
) -> Any:
    prompt = ChatPromptTemplate.from_template(_read_text(prompt_path)).partial(
        requests_table=requests_table,
        product_context=product_context,
    )
    return prompt | llm.with_structured_output(JudgeArrayAnswer)


def run_stage1_classification(
    source_df: pd.DataFrame,
    chain: Any,
    output_dir: str | Path,
    text_column: str = "description_claim",
    batch_size: int = 100,
    max_concurrency: int = 5,
    retries: int = 4,
    base_sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    out_dir = _ensure_dir(output_dir)
    batches: list[pd.DataFrame] = []
    for batch_idx, temp_df in _iter_batches(source_df, batch_size):
        chunk_path = out_dir / f"stage1_classification_{batch_idx:04d}.parquet"
        if chunk_path.exists():
            batches.append(pd.read_parquet(chunk_path))
            continue

        payloads = [{"text": str(text)} for text in temp_df[text_column].fillna("")]
        answers = _batch_with_backoff(
            chain,
            payloads,
            max_concurrency=max_concurrency,
            retries=retries,
            base_sleep_seconds=base_sleep_seconds,
        )
        parsed = [_to_plain_dict(answer, ClassificationAnswer()) for answer in answers]
        temp_df["issues_pred"] = [item.get("issues", []) for item in parsed]
        temp_df["requested_actions_pred"] = [item.get("requested_actions", []) for item in parsed]
        temp_df.to_parquet(chunk_path, index=False)
        batches.append(temp_df)
    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()


def run_judge(
    classified_df: pd.DataFrame,
    chain: Any,
    output_dir: str | Path,
    labels_column: str,
    text_column: str = "description_claim",
    batch_size: int = 100,
    max_concurrency: int = 5,
    retries: int = 4,
    base_sleep_seconds: float = 2.0,
    file_prefix: str = "judge",
) -> pd.DataFrame:
    out_dir = _ensure_dir(output_dir)
    batches: list[pd.DataFrame] = []
    for batch_idx, temp_df in _iter_batches(classified_df, batch_size):
        chunk_path = out_dir / f"{file_prefix}_{batch_idx:04d}.parquet"
        if chunk_path.exists():
            batches.append(pd.read_parquet(chunk_path))
            continue

        payloads = []
        label_counts = []
        for _, row in temp_df.iterrows():
            labels = _classification_labels_to_list(row.get(labels_column))
            payloads.append(
                {
                    "text": str(row.get(text_column, "")),
                    "labels": json.dumps(labels, ensure_ascii=False),
                }
            )
            label_counts.append(len(labels))

        answers = _batch_with_backoff(
            chain,
            payloads,
            max_concurrency=max_concurrency,
            retries=retries,
            base_sleep_seconds=base_sleep_seconds,
        )

        rows: list[dict[str, Any]] = []
        for row_idx, (answer, count) in enumerate(zip(answers, label_counts)):
            verdicts = _parse_judge_verdicts(answer, count)
            orig_row = temp_df.iloc[row_idx]
            for verdict in verdicts:
                rows.append(
                    {
                        "row_id": temp_df.index[row_idx],
                        "text": str(orig_row.get(text_column, "")),
                        "category": verdict.get("category", ""),
                        "sub_category": verdict.get("sub_category", ""),
                        "judge_category_ok": bool(verdict.get("judge_category_ok", False)),
                        "judge_sub_category_ok": bool(verdict.get("judge_sub_category_ok", False)),
                        "suggested_category": verdict.get("suggested_category", ""),
                        "suggested_sub_category": verdict.get("suggested_sub_category", ""),
                    }
                )

        chunk_df = pd.DataFrame(rows)
        chunk_df.to_parquet(chunk_path, index=False)
        batches.append(chunk_df)
    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()


def _make_str_enum(name: str, values: Sequence[str]) -> type:
    members = {f"v{i}": value for i, value in enumerate(values)}
    return Enum(name, members, type=str)


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _iter_batches(df: pd.DataFrame, batch_size: int):
    for start in range(0, len(df), batch_size):
        yield start // batch_size, df.iloc[start : start + batch_size].copy()


def _batch_with_backoff(
    runnable: Any,
    payloads: Sequence[dict[str, Any]],
    max_concurrency: int = 5,
    retries: int = 4,
    base_sleep_seconds: float = 2.0,
) -> list[Any]:
    payloads = list(payloads)
    if not payloads:
        return []
    try:
        return runnable.batch(payloads, config={"max_concurrency": max_concurrency})
    except Exception:
        return [
            _safe_invoke(runnable, payload, retries, base_sleep_seconds)
            for payload in payloads
        ]


def _safe_invoke(
    runnable: Any,
    payload: dict[str, Any],
    retries: int,
    base_sleep_seconds: float,
) -> Any:
    for attempt in range(retries):
        try:
            return runnable.invoke(payload)
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(base_sleep_seconds * (2**attempt))
    return None


def _to_plain_dict(item: Any, fallback: BaseModel) -> dict[str, Any]:
    if item is None:
        return fallback.model_dump(mode="json")
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    if isinstance(item, dict):
        return item
    return fallback.model_dump(mode="json")


def _classification_labels_to_list(value: Any) -> list[dict[str, Any]]:
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
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return _classification_labels_to_list(json.loads(value))
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        return [value]
    return []


def _parse_judge_verdicts(answer: Any, expected_count: int) -> list[dict[str, Any]]:
    fallback = {
        "category": "",
        "sub_category": "",
        "judge_category_ok": False,
        "judge_sub_category_ok": False,
        "suggested_category": "",
        "suggested_sub_category": "",
    }
    if answer is None:
        return [fallback.copy() for _ in range(expected_count)]
    if isinstance(answer, JudgeArrayAnswer):
        verdicts = [item.model_dump(mode="json") for item in answer.verdicts]
    elif isinstance(answer, dict) and "verdicts" in answer:
        verdicts = answer["verdicts"]
    elif hasattr(answer, "model_dump"):
        verdicts = answer.model_dump(mode="json").get("verdicts", [])
    else:
        return [fallback.copy() for _ in range(expected_count)]
    while len(verdicts) < expected_count:
        verdicts.append(fallback.copy())
    return verdicts[:expected_count]
