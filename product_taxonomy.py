"""One product config with all themes and subclasses; no issue/request split."""
from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, create_model

from llm_pipeline import _batch_with_backoff, _make_str_enum
from utils.result_builder import RESULT_COLUMNS

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = """Ты — аналитик обращений банка. Выбери подклассы обращения.
Продукт: {product}
Тема: {theme}
Продукт и тема заданы БД. Не определяй и не меняй их.
Контекст продукта: {product_context}
Допустимые подклассы только для этой темы (subclass, description):
{taxonomy}
Верни объект с массивом labels. Каждая метка содержит subclass (точное название
из справочника) и description (краткое обоснование по тексту).
Не разделяй проблемы и запросы: это единая классификация обращения.
Можно выбрать несколько подклассов, если текст содержит несколько аспектов.
Не повторяй подкласс. Если подходящего подкласса нет или текст служебный,
не содержит сути обращения, верни пустой labels. Не выдумывай содержание.
Текст обращения:
{text}
"""

JUDGE_PROMPT = """Проверь назначенные подклассы обращения.
Продукт: {product}
Тема из БД: {theme}
Продукт и тему не оценивай и не меняй.
Контекст продукта: {product_context}
Допустимые подклассы:
{taxonomy}
Для каждой входной метки повтори subclass и верни ok: true, только если
подкласс соответствует тексту и этому справочнику, иначе false.
Верни объект с массивом verdicts в том же порядке и той же длины, что labels.
Текст обращения:
{text}
Метки:
{labels}
"""


class SubclassVerdict(BaseModel):
    subclass: str
    ok: bool


class JudgeAnswer(BaseModel):
    verdicts: list[SubclassVerdict]


def normalize_key(value: str) -> str:
    """Ignore whitespace artifacts, including NBSP/newlines; keep case significant."""
    return " ".join(value.split())


def taxonomy_by_theme(config: dict) -> dict[str, list[dict[str, str]]]:
    product = config.get("product")
    if not isinstance(product, str) or not normalize_key(product):
        raise ValueError("product must be a non-empty string")
    if config.get("theme") is not None or config.get("category") is not None:
        raise ValueError("product_taxonomy uses all themes: omit theme/category filters")
    items = config.get("taxonomy")
    if not isinstance(items, list) or not items:
        raise ValueError("taxonomy must be a non-empty array")
    groups: dict[str, list[dict[str, str]]] = {}
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("taxonomy entries must be objects")
        for field in ("theme", "subclass", "description"):
            if not isinstance(item.get(field), str) or not normalize_key(item[field]):
                raise ValueError(f"taxonomy.{field} must be a non-empty string")
        theme = normalize_key(item["theme"])
        subclass = normalize_key(item["subclass"])
        if (theme, subclass) in seen:
            raise ValueError(f"Duplicate subclass in theme {theme!r}: {subclass!r}")
        seen.add((theme, subclass))
        groups.setdefault(theme, []).append({
            "subclass": subclass, "description": item["description"].strip(),
        })
    return groups


def build_subclass_model(items: list[dict[str, str]]) -> type[BaseModel]:
    enum = _make_str_enum("SubclassEnum", [item["subclass"] for item in items])
    label = create_model("ProductLabel", subclass=(enum, ...), description=(str, ...))
    return create_model("ProductAnswer", labels=(list[label], Field(default_factory=list)))


def _parse(answer: Any, model: type[BaseModel]) -> dict:
    if answer is None:
        return {}
    try:
        if isinstance(answer, BaseModel):
            answer = answer.model_dump(mode="json")
        return model.model_validate(answer).model_dump(mode="json")
    except (ValueError, TypeError):
        logger.warning("Invalid %s answer; treating as failed", model.__name__)
        return {}


def score_product_taxonomy(
    source_df: pd.DataFrame, *, llm: Any, config: dict, product_context: str,
    text_column: str, report_date: str, classifier_name: str,
) -> pd.DataFrame:
    groups = taxonomy_by_theme(config)
    missing = {"product", "theme", text_column} - set(source_df.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    runtime = config.get("runtime", {})
    batch_size = runtime.get("batch_size", 100)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    options = dict(
        max_concurrency=runtime.get("max_concurrency", 5),
        retries=runtime.get("retries", 4),
        base_sleep_seconds=runtime.get("backoff_base_seconds", 2.0),
    )
    product_mask = source_df["product"].map(
        lambda value: isinstance(value, str) and value == config["product"]
    )
    source = source_df.loc[product_mask].copy()
    theme_keys = source["theme"].map(
        lambda value: normalize_key(value) if isinstance(value, str) else ""
    )
    unknown = ~theme_keys.isin(groups)
    if unknown.any():
        logger.warning(
            "%s: skipped %s rows with unknown/missing Theme: %s",
            classifier_name, int(unknown.sum()),
            sorted(set(theme_keys[unknown])),
        )
    rows = []
    for theme, items in groups.items():
        data = source.loc[theme_keys == theme].reset_index(drop=True)
        if data.empty:
            continue
        partial = dict(
            product=config["product"], theme=theme, product_context=product_context,
            taxonomy=json.dumps(items, ensure_ascii=False),
        )
        model = build_subclass_model(items)
        chain = ChatPromptTemplate.from_template(CLASSIFY_PROMPT).partial(**partial) | llm.with_structured_output(model)
        judge = ChatPromptTemplate.from_template(JUDGE_PROMPT).partial(**partial) | llm.with_structured_output(JudgeAnswer)
        for start in range(0, len(data), batch_size):
            batch = data.iloc[start:start + batch_size]
            texts = [str(value) for value in batch[text_column].fillna("")]
            answers = _batch_with_backoff(chain, [{"text": value} for value in texts], **options)
            parsed = [_parse(answer, model).get("labels", []) for answer in answers]
            # One result per selected subclass, even if an endpoint repeats a label.
            labels = [list({label["subclass"]: label for label in group}.values()) for group in parsed]
            active = [i for i, group in enumerate(labels) if group]
            judge_answers = _batch_with_backoff(judge, [
                {"text": texts[i], "labels": json.dumps(labels[i], ensure_ascii=False)}
                for i in active
            ], **options)
            verdicts = {
                i: _parse(answer, JudgeAnswer).get("verdicts", [])
                for i, answer in zip(active, judge_answers)
            }
            for i, (_, source_row) in enumerate(batch.iterrows()):
                checks = verdicts.get(i, [])
                for j, label in enumerate(labels[i]):
                    check = checks[j] if j < len(checks) else {}
                    rows.append({
                        "claim_num": source_row.get("claim_num", ""),
                        "created": source_row.get("created"),
                        "report_date": pd.to_datetime(report_date),
                        "product": source_row["product"],
                        "theme": source_row["theme"],
                        "category": source_row.get("category"),
                        "classifier_name": classifier_name,
                        "type": "subclass",
                        "class": None,
                        "sub_class": label["subclass"],
                        "description": label["description"],
                        "eval": check.get("subclass") == label["subclass"] and bool(check.get("ok", False)),
                    })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
