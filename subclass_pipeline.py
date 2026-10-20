"""Optional subclass-only scoring; existing full classification is unchanged."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, create_model

from llm_pipeline import _make_str_enum, run_judge, run_stage1_classification
from utils.result_builder import build_result_frame


CLASSIFICATION_PROMPT = """Ты — аналитик жалоб банка. Определи только подклассы по тексту.
Продукт и тема уже заданы источником данных; не определяй их заново.
Контекст: {product_context}
Справочник проблем issues (sub_category, description):
{issues_table}
Справочник запросов requested_actions (sub_category, description):
{requests_table}
Для каждой метки верни sub_category строго из соответствующего справочника
и description — краткое обоснование по тексту, а не копию справочника.
Не выдумывай проблемы и запросы. Для отсутствующих проблем или запросов
верни пустой массив. Для служебного текста без сути верни пустые массивы.
Одна метка на подкласс в каждом массиве. Причины жалобы относятся к issues,
явные просьбы клиента — к requested_actions.
Верни объект с массивами issues и requested_actions.
Текст обращения:
{text}
"""

JUDGE_PROMPT = """Проверь назначенные подклассы {label_type} по тексту обращения.
Продукт и тема берутся из БД и не подлежат проверке.
Контекст: {product_context}
Справочник подклассов:
{taxonomy}
Для каждой входной метки повтори sub_category и верни
judge_sub_category_ok: true, если подкласс соответствует тексту и справочнику,
иначе false. В suggested_sub_category укажи более подходящий подкласс
из справочника либо пустую строку, если исправление не требуется или неясно.
Верни объект с массивом verdicts той же длины и в том же порядке, что labels.
Текст обращения:
{text}
Метки:
{labels}
"""


class SubclassVerdict(BaseModel):
    sub_category: str
    judge_sub_category_ok: bool
    suggested_sub_category: str = ""


class SubclassJudgeAnswer(BaseModel):
    verdicts: list[SubclassVerdict]


def load_taxonomy(path: str | Path) -> list[dict[str, str]]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: expected a non-empty taxonomy array")
    result = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: taxonomy entries must be objects")
        name = item.get("sub_category")
        description = item.get("description")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}: sub_category must be a non-empty string")
        if not isinstance(description, str):
            raise ValueError(f"{path}: description must be a string")
        if name in seen:
            raise ValueError(f"{path}: duplicate sub_category {name!r}")
        seen.add(name)
        result.append({"sub_category": name, "description": description})
    return result


def build_subclass_model(issues, requests) -> type[BaseModel]:
    def label_model(name, items):
        enum = _make_str_enum(name + "Enum", [item["sub_category"] for item in items])
        return create_model(name, sub_category=(enum, ...), description=(str, ...))

    issue_label = label_model("IssueSubclass", issues)
    request_label = label_model("RequestSubclass", requests)
    return create_model(
        "SubclassAnswer",
        issues=(list[issue_label], Field(default_factory=list)),
        requested_actions=(list[request_label], Field(default_factory=list)),
    )


def score_subclasses(
    source_df: pd.DataFrame, *, llm: Any, issues_path: Path, requests_path: Path,
    product_context: str, text_column: str, runtime: dict,
    report_date: str, classifier_name: str,
) -> pd.DataFrame:
    issues = load_taxonomy(issues_path)
    requests = load_taxonomy(requests_path)
    issues_table = json.dumps(issues, ensure_ascii=False)
    requests_table = json.dumps(requests, ensure_ascii=False)
    prompt = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT).partial(
        product_context=product_context,
        issues_table=issues_table, requests_table=requests_table,
    )
    chain = prompt | llm.with_structured_output(build_subclass_model(issues, requests))
    options = dict(
        text_column=text_column,
        batch_size=runtime.get("batch_size", 100),
        max_concurrency=runtime.get("max_concurrency", 5),
        retries=runtime.get("retries", 4),
        base_sleep_seconds=runtime.get("backoff_base_seconds", 2.0),
    )
    classified = run_stage1_classification(source_df, chain, **options)
    judges = []
    for column, label_type, taxonomy in (
        ("issues_pred", "issues", issues_table),
        ("requested_actions_pred", "requested_actions", requests_table),
    ):
        judge_prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT).partial(
            product_context=product_context, label_type=label_type, taxonomy=taxonomy,
        )
        judge_chain = judge_prompt | llm.with_structured_output(SubclassJudgeAnswer)
        verdicts = run_judge(classified, judge_chain, labels_column=column, **options)
        # The shared builder combines both flags; only the subclass flag matters here.
        verdicts["judge_category_ok"] = True
        judges.append(verdicts)
    result = build_result_frame(
        classified, *judges, report_date=report_date, classifier_name=classifier_name,
    )
    # SQL writer converts None into SQL NULL. Source product/theme/category stay intact.
    result["class"] = None
    return result
