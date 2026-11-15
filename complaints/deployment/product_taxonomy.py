"""Theme-scoped product classification and a shared catalog of all products."""
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
UNCLASSIFIED_ID = "unclassified"

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


CATALOG_CLASSIFY_PROMPT = """Ты — аналитик обращений банка.
Продукт задан БД: {product}. Выбирай подтипы только этого продукта.
Контекст продукта: {product_context}
Действующий справочник (entry_id, subclass, description, keywords):
{taxonomy}
По тексту найди все отдельные проблемы, максимум 10 на обращение.
Выбирай конкретные записи по описаниям. Одинаковые названия подтипов могут
иметь разные описания. Не объединяй их и не повторяй одну запись.
Для одной и той же проблемы выбирай одну запись с наиболее точным описанием.
Не разделяй проблемы и запросы. Верни labels с entry_id выбранной записи
и description — кратким обоснованием по тексту обращения.
Если подходящей записи нет или текст служебный и не содержит сути,
верни labels с одной меткой entry_id="unclassified" и description с причиной,
почему обращение не классифицировано. Не выдумывай содержание или новые подтипы.
Если текст содержит только «Во вложении», «См. вложение», «Входящий документ…»,
«Входной документ…», номер документа или шаблон без описания самой проблемы,
верни одну метку entry_id="unclassified" и укажи, что в тексте отсутствует суть жалобы.
Содержимое вложений тебе не передано.
Не восстанавливай суть жалобы по названию продукта, номеру документа или справочнику.
Если вместе со ссылкой на вложение описана проблема, классифицируй изложенную проблему.
Массив labels должен содержать хотя бы одну метку. Не совмещай unclassified
с метками подтипов из справочника.
Текст обращения:
{text}
"""

CATALOG_JUDGE_PROMPT = """Проверь назначенные подтипы обращения.
Продукт задан БД: {product}.
Контекст продукта: {product_context}
Действующий справочник этого продукта:
{taxonomy}
Для каждой метки проверь соответствие текста описанию конкретной записи.
Если текст содержит только отсылку к вложению, «Входящий документ…»,
«Входной документ…», номер или служебный шаблон без сути жалобы,
верни ok: false для всех меток. Содержимое вложений тебе не передано.
Если вместе со ссылкой на вложение описана проблема, проверяй метки по её описанию.
Верни verdicts с entry_id проверяемой записи и ok: true или false.
Каждый входной entry_id должен встретиться ровно один раз.
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


class CatalogVerdict(BaseModel):
    entry_id: str
    ok: bool


class CatalogJudgeAnswer(BaseModel):
    verdicts: list[CatalogVerdict]


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


def taxonomy_by_product(config: dict) -> dict[str, list[dict[str, str]]]:
    """Validate the editable catalog and assign IDs within each product."""
    if any(config.get(field) is not None for field in ("product", "theme", "category")):
        raise ValueError("Product catalog requires product=null and no theme/category filters")
    items = config.get("taxonomy")
    if not isinstance(items, list) or not items:
        raise ValueError("taxonomy must be a non-empty array")
    groups: dict[str, list[dict[str, str]]] = {}
    seen = set()
    for row_number, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"taxonomy row {row_number} must be an object")
        for field in ("product", "theme", "subclass", "description"):
            if not isinstance(item.get(field), str) or not normalize_key(item[field]):
                raise ValueError(f"taxonomy row {row_number}: {field} must be a non-empty string")
        product = normalize_key(item["product"])
        theme = normalize_key(item["theme"])
        subclass = normalize_key(item["subclass"])
        if (product, theme, subclass) in seen:
            raise ValueError(f"Duplicate taxonomy row {row_number}: {(product, theme, subclass)!r}")
        seen.add((product, theme, subclass))
        keywords = item.get("keywords", "")
        if not isinstance(keywords, str):
            raise ValueError(f"taxonomy row {row_number}: keywords must be a string")
        product_items = groups.setdefault(product, [])
        product_items.append(dict(
            entry_id=f"r{len(product_items) + 1}", theme=theme, subclass=subclass,
            description=item["description"].strip(), keywords=keywords.strip(),
        ))
    return groups


def build_catalog_model(items: list[dict[str, str]]) -> type[BaseModel]:
    enum = _make_str_enum("CatalogEntryEnum", [item["entry_id"] for item in items] + [UNCLASSIFIED_ID])
    label = create_model("CatalogLabel", entry_id=(enum, ...), description=(str, ...))
    return create_model("CatalogAnswer", labels=(list[label], Field(..., min_length=1, max_length=10)))


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


def unclassified_result_row(
    source_row: pd.Series, *, report_date: str, classifier_name: str, description: str,
) -> dict:
    """Сохранить обращение и причину, по которой ему не назначен подтип."""
    return {
        "claim_num": source_row.get("claim_num", ""),
        "created": source_row.get("created"),
        "report_date": pd.to_datetime(report_date),
        "product": source_row.get("product"),
        "theme": source_row.get("theme"),
        "category": source_row.get("category"),
        "classifier_name": classifier_name,
        "type": "subclass",
        "class": None,
        "sub_class": None,
        "description": description,
        "eval": False,
    }


def score_product_taxonomy(
    source_df: pd.DataFrame, *, llm: Any, config: dict, product_context: str,
    text_column: str, report_date: str, classifier_name: str,
    catalog_items: list[dict[str, str]] | None = None,
    run_logger: Any = None,
    diagnostics: dict[str, int] | None = None,
) -> pd.DataFrame:
    catalog_mode = catalog_items is not None
    run_logger = run_logger or logger
    if diagnostics is None:
        diagnostics = dict(classified=0, no_labels=0, no_response=0, invalid_answer=0)
    skip_reasons = {
        "no_labels": "модель не выбрала ни одного подтипа",
        "no_response": "после повторов не получен ответ LLM (сбой вызова или разбора ответа)",
        "invalid_answer": "ответ не прошёл проверку формата, допустимых ID или лимита 10 меток",
    }
    groups = {"": catalog_items} if catalog_mode else taxonomy_by_theme(config)
    required = {"product", text_column} if catalog_mode else {"product", "theme", text_column}
    missing = required - set(source_df.columns)
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
    themes = source.get("theme", pd.Series(index=source.index, dtype=object))
    theme_keys = themes.map(
        lambda value: normalize_key(value) if isinstance(value, str) else ""
    )
    unknown = ~theme_keys.isin(groups)
    if not catalog_mode and unknown.any():
        logger.warning(
            "%s: skipped %s rows with unknown/missing Theme: %s",
            classifier_name, int(unknown.sum()),
            sorted(set(theme_keys[unknown])),
        )
    rows = []
    for theme, items in groups.items():
        data = (source if catalog_mode else source.loc[theme_keys == theme]).reset_index(drop=True)
        if data.empty:
            continue
        label_key = "entry_id" if catalog_mode else "subclass"
        items_by_id = {item["entry_id"]: item for item in items} if catalog_mode else {}
        # The business type is used only for the final lookup, never as LLM input.
        prompt_items = [
            {key: value for key, value in item.items() if key != "theme"}
            for item in items
        ] if catalog_mode else items
        partial = dict(
            product=config["product"], theme=theme, product_context=product_context,
            taxonomy=json.dumps(prompt_items, ensure_ascii=False),
        )
        model = build_catalog_model(items) if catalog_mode else build_subclass_model(items)
        judge_model = CatalogJudgeAnswer if catalog_mode else JudgeAnswer
        classify_prompt = CATALOG_CLASSIFY_PROMPT if catalog_mode else CLASSIFY_PROMPT
        judge_prompt = CATALOG_JUDGE_PROMPT if catalog_mode else JUDGE_PROMPT
        chain = ChatPromptTemplate.from_template(classify_prompt).partial(**partial) | llm.with_structured_output(model)
        judge = ChatPromptTemplate.from_template(judge_prompt).partial(**partial) | llm.with_structured_output(judge_model)
        for start in range(0, len(data), batch_size):
            batch = data.iloc[start:start + batch_size]
            texts = [str(value) for value in batch[text_column].fillna("")]
            answers = _batch_with_backoff(chain, [{"text": value} for value in texts], **options)
            parsed_answers = [_parse(answer, model) for answer in answers]
            parsed = [answer.get("labels", []) for answer in parsed_answers]
            # One result per selected subclass, even if an endpoint repeats a label.
            labels = [list({label[label_key]: label for label in group}.values()) for group in parsed]
            if catalog_mode:
                for i, group in enumerate(labels):
                    classified = [label for label in group if label["entry_id"] != UNCLASSIFIED_ID]
                    labels[i] = classified or group
                active = [
                    i for i, group in enumerate(labels)
                    if group and group[0]["entry_id"] != UNCLASSIFIED_ID
                ]
            else:
                active = [i for i, group in enumerate(labels) if group]
            judge_answers = _batch_with_backoff(judge, [
                {"text": texts[i], "labels": json.dumps(labels[i], ensure_ascii=False)}
                for i in active
            ], **options)
            verdicts = {
                i: _parse(answer, judge_model).get("verdicts", [])
                for i, answer in zip(active, judge_answers)
            }
            for i, (_, source_row) in enumerate(batch.iterrows()):
                if catalog_mode:
                    reason = None
                    if answers[i] is None:
                        reason = "no_response"
                    elif not parsed_answers[i]:
                        reason = "invalid_answer"
                    elif not labels[i]:
                        reason = "no_labels"
                    elif labels[i][0]["entry_id"] == UNCLASSIFIED_ID:
                        reason = "no_labels"
                    if reason is None:
                        diagnostics["classified"] += 1
                    else:
                        diagnostics[reason] += 1
                        description = skip_reasons[reason]
                        if reason == "no_labels" and labels[i]:
                            description = labels[i][0]["description"].strip() or description
                        run_logger.warning(
                            "%s: претензия %s, продукт=%r — подтип не назначен: %s",
                            classifier_name, source_row.get("claim_num", "не указан"),
                            source_row["product"], description,
                        )
                        rows.append(unclassified_result_row(
                            source_row, report_date=report_date,
                            classifier_name=classifier_name, description=description,
                        ))
                        continue
                checks = verdicts.get(i, [])
                for j, label in enumerate(labels[i]):
                    check = checks[j] if j < len(checks) else {}
                    item = items_by_id[label["entry_id"]] if catalog_mode else {}
                    if catalog_mode:
                        matching = [check for check in checks if check["entry_id"] == label["entry_id"]]
                        check = matching[0] if len(matching) == 1 else {}
                    rows.append({
                        "claim_num": source_row.get("claim_num", ""),
                        "created": source_row.get("created"),
                        "report_date": pd.to_datetime(report_date),
                        "product": source_row["product"],
                        "theme": source_row.get("theme"),
                        "category": source_row.get("category"),
                        "classifier_name": classifier_name,
                        "type": "subclass",
                        "class": item.get("theme"),
                        "sub_class": item["subclass"] if catalog_mode else label["subclass"],
                        "description": label["description"],
                        "eval": check.get(label_key) == label[label_key] and bool(check.get("ok", False)),
                    })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
