from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Ограничение из ТЗ: HTTP timeout не должен превышать 60 секунд.
MAX_HTTP_TIMEOUT_SECONDS = 60


# =========================
# Pydantic-схемы (контракты)
# =========================
# Эти классы описывают структуру данных, которые мы хотим получать от модели.
# Когда используем llm.with_structured_output(...), LangChain "подсказывает"
# модели нужный JSON-формат и валидирует ответ под эту схему.


class Stage0Item(BaseModel):
    """Один элемент из stage 0: тип + описание проблемы/запроса."""

    type: str
    description: str


class Stage0Extraction(BaseModel):
    """Результат извлечения сущностей из текста обращения."""

    issues: list[Stage0Item] = Field(default_factory=list)
    requested_actions: list[Stage0Item] = Field(default_factory=list)


class ClassificationLabel(BaseModel):
    """Одна разметка из справочника (category/sub_category/description)."""

    category: str
    sub_category: str
    description: str


class ClassificationAnswer(BaseModel):
    """Ответ этапа 1: классификация проблем и запросов клиента."""

    issues: list[ClassificationLabel] = Field(default_factory=list)
    requested_actions: list[ClassificationLabel] = Field(default_factory=list)


class JudgeAnswer(BaseModel):
    """Ответ judge-цепочки: корректны ли категория и подкатегория."""

    category: bool
    sub_category: bool


class LegacyJudgeAnswer(BaseModel):
    """Ответ legacy judge для старого плоского справочника."""

    decision: bool


class TaxonomyItem(BaseModel):
    """Одна строка таксономии: category / sub_category / description."""

    category: str
    sub_category: str
    description: str


class TaxonomyResult(BaseModel):
    """Результат этапа 0b: список элементов таксономии."""

    items: list[TaxonomyItem] = Field(default_factory=list)


# =========================
# Конфиг проекта (JSON)
# =========================


class LLMConfig(BaseModel):
    """Настройки подключения к LLM.

    Важно:
    - timeout_seconds автоматически ограничивается 60 секундами;
    - все поля читаются из обычного JSON-файла, не из переменных окружения.
    """

    api_key: str
    base_url: str
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    timeout_seconds: int = MAX_HTTP_TIMEOUT_SECONDS
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


class RuntimeConfig(BaseModel):
    """Параметры массовой обработки (батчи/ретраи/параллелизм)."""

    batch_size: int = 100
    max_concurrency: int = 5
    retries: int = 4
    backoff_base_seconds: float = 2.0


class PairConfig(BaseModel):
    """Полный конфиг для пары (product_id, product_category)."""

    product_id: str
    product_category: str
    input_path: Path
    prompts_dir: Path
    taxonomy_issues_path: Path
    taxonomy_requests_path: Path
    text_column: str = "text"
    llm: LLMConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @property
    def pair_key(self) -> str:
        """Удобный ключ пары для именования папок результатов.

        Возвращает 'product_id/product_category', чтобы все категории
        одного продукта хранились в одной родительской папке.
        """
        return f"{self.product_id}/{self.product_category}"


def _ensure_path(path: str | Path) -> Path:
    """Нормализует str/Path в pathlib.Path."""
    return path if isinstance(path, Path) else Path(path)


def _ensure_dir(path: str | Path) -> Path:
    """Создает директорию (если ее нет) и возвращает Path."""
    p = _ensure_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Разрешает относительные пути относительно директории конфига."""
    p = _ensure_path(path)
    if p.is_absolute() or base_dir is None:
        return p.resolve()
    return (_ensure_path(base_dir) / p).resolve()


def load_pair_config(config_path: str | Path) -> PairConfig:
    """Читает JSON-конфиг и приводит пути к абсолютным.

    Почему это важно:
    - notebook можно запускать из разных директорий;
    - пути в конфиге остаются короткими/читаемыми (например ../prompts/...).
    """

    config_path = _ensure_path(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = PairConfig(**payload)
    base_dir = config_path.parent
    cfg.input_path = _resolve_path(cfg.input_path, base_dir=base_dir)
    cfg.prompts_dir = _resolve_path(cfg.prompts_dir, base_dir=base_dir)
    cfg.taxonomy_issues_path = _resolve_path(cfg.taxonomy_issues_path, base_dir=base_dir)
    cfg.taxonomy_requests_path = _resolve_path(cfg.taxonomy_requests_path, base_dir=base_dir)
    return cfg


def read_text(path: str | Path) -> str:
    """Читает UTF-8 текст из файла."""
    return _ensure_path(path).read_text(encoding="utf-8")


def read_product_context(prompts_dir: str | Path) -> str:
    """Читает product_context.txt из папки промптов.

    Если файла нет — возвращает пустую строку (промпты будут работать без контекста).
    """
    ctx_path = _ensure_path(prompts_dir) / "product_context.txt"
    if ctx_path.exists():
        return ctx_path.read_text(encoding="utf-8").strip()
    return ""


def read_taxonomy_tables(config: PairConfig) -> tuple[str, str]:
    """Загружает таблицы справочников issues/requested_actions как строки."""
    return read_text(config.taxonomy_issues_path), read_text(config.taxonomy_requests_path)


def build_llm_from_config(llm_config: LLMConfig) -> ChatOpenAI:
    """Создает LLM-клиент из JSON-конфига.

    LangChain-идея:
    - ChatOpenAI здесь выступает как универсальный чат-клиент;
    - позже этот объект подключается к prompt-шаблонам через оператор `|`.
    """

    timeout_seconds = min(llm_config.timeout_seconds, MAX_HTTP_TIMEOUT_SECONDS)
    return ChatOpenAI(
        model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        temperature=llm_config.temperature,
        timeout=timeout_seconds,
        max_retries=0,  # ретраи мы контролируем самостоятельно через backoff-функции ниже
        model_kwargs=llm_config.model_kwargs,
    )


# =========================
# Надежные вызовы LLM
# =========================


def invoke_with_backoff(
    runnable: Any,
    payload: dict[str, Any],
    retries: int = 4,
    base_sleep_seconds: float = 2.0,
) -> Any:
    """Одиночный вызов chain.invoke(...) с экспоненциальным backoff.

    Формула ожидания между повторами:
    base_sleep_seconds * (2 ** attempt)
    """

    for attempt in range(retries):
        try:
            return runnable.invoke(payload)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(base_sleep_seconds * (2**attempt))
    raise RuntimeError("Unreachable state in invoke_with_backoff.")


def batch_with_backoff(
    runnable: Any,
    payloads: Sequence[dict[str, Any]],
    max_concurrency: int = 5,
    retries: int = 4,
    base_sleep_seconds: float = 2.0,
) -> list[Any]:
    """Пакетный вызов chain.batch(...), а при ошибке — fallback на invoke по одному.

    Зачем fallback:
    - если batch падает из-за нестабильности API, мы все равно обрабатываем записи;
    - теряем скорость, но повышаем устойчивость пайплайна.
    """

    payloads = list(payloads)
    if not payloads:
        return []
    try:
        return runnable.batch(payloads, config={"max_concurrency": max_concurrency})
    except Exception:
        return [
            invoke_with_backoff(
                runnable,
                payload,
                retries=retries,
                base_sleep_seconds=base_sleep_seconds,
            )
            for payload in payloads
        ]


# =========================
# Конструкторы LCEL-цепочек
# =========================
# LCEL = LangChain Expression Language.
# Базовая форма: prompt | llm
# Где:
# - prompt подставляет переменные в текст;
# - llm выполняет запрос.


def build_stage0_extract_chain(
    llm: ChatOpenAI,
    prompt_path: str | Path,
    product_context: str = "",
) -> Any:
    """Этап 0a: извлекаем issues/requested_actions из текста обращения."""
    prompt = ChatPromptTemplate.from_template(read_text(prompt_path)).partial(
        product_context=product_context,
    )
    return prompt | llm.with_structured_output(Stage0Extraction)


def build_stage0_taxonomy_chain(
    llm: ChatOpenAI,
    prompt_path: str | Path,
    product_context: str = "",
) -> Any:
    """Этап 0b: агрегируем проблемы в двухуровневую таксономию."""
    prompt = ChatPromptTemplate.from_template(read_text(prompt_path)).partial(
        product_context=product_context,
    )
    return prompt | llm.with_structured_output(TaxonomyResult)


def build_stage1_chain(
    llm: ChatOpenAI,
    prompt_path: str | Path,
    issues_table: str,
    requests_table: str,
    product_context: str = "",
) -> Any:
    """Этап 1: классифицируем обращение по утвержденному справочнику.

    `.partial(...)` подставляет "постоянные" переменные один раз.
    После этого на каждый вызов остается передавать только `text`.
    """

    prompt = ChatPromptTemplate.from_template(read_text(prompt_path)).partial(
        issues_table=issues_table,
        requests_table=requests_table,
        product_context=product_context,
    )
    return prompt | llm.with_structured_output(ClassificationAnswer)


def build_judge_issues_chain(
    llm: ChatOpenAI,
    prompt_path: str | Path,
    issues_table: str,
    product_context: str = "",
) -> Any:
    """Judge-цепочка для проверки корректности разметки issues."""
    prompt = ChatPromptTemplate.from_template(read_text(prompt_path)).partial(
        issues_table=issues_table,
        product_context=product_context,
    )
    return prompt | llm.with_structured_output(JudgeAnswer)


def build_judge_requests_chain(
    llm: ChatOpenAI,
    prompt_path: str | Path,
    requests_table: str,
    product_context: str = "",
) -> Any:
    """Judge-цепочка для проверки корректности разметки requested_actions."""
    prompt = ChatPromptTemplate.from_template(read_text(prompt_path)).partial(
        requests_table=requests_table,
        product_context=product_context,
    )
    return prompt | llm.with_structured_output(JudgeAnswer)


def build_judge_legacy_chain(llm: ChatOpenAI, prompt_path: str | Path) -> Any:
    """Опциональная judge-цепочка для старого плоского справочника."""
    prompt = ChatPromptTemplate.from_template(read_text(prompt_path))
    return prompt | llm.with_structured_output(LegacyJudgeAnswer)


def _to_plain_dict(item: Any, fallback: BaseModel) -> dict[str, Any]:
    """Преобразует результат chain в dict, даже если формат ответа нестабилен."""
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return item
    return fallback.model_dump()


def _iter_batches(df: pd.DataFrame, batch_size: int) -> Iterator[tuple[int, pd.DataFrame]]:
    """Итератор по батчам DataFrame с индексом батча."""
    for batch_idx, start in enumerate(range(0, len(df), batch_size)):
        yield batch_idx, df.iloc[start : start + batch_size].copy()


# =========================
# Этап 0: извлечение и таксономия
# =========================


def run_stage0_extract(
    source_df: pd.DataFrame,
    chain: Any,
    output_dir: str | Path,
    text_column: str = "text",
    batch_size: int = 100,
    max_concurrency: int = 5,
    retries: int = 4,
    base_sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    """Запускает этап 0a батчами и сохраняет каждый батч в parquet.

    Выходные колонки:
    - issues_raw
    - requested_actions_raw
    """

    out_dir = _ensure_dir(output_dir)
    batches: list[pd.DataFrame] = []
    for batch_idx, temp_df in _iter_batches(source_df, batch_size):
        payloads = [{"text": str(text)} for text in temp_df[text_column].fillna("")]
        answers = batch_with_backoff(
            chain,
            payloads,
            max_concurrency=max_concurrency,
            retries=retries,
            base_sleep_seconds=base_sleep_seconds,
        )
        parsed = [_to_plain_dict(a, Stage0Extraction()) for a in answers]

        temp_df["issues_raw"] = [p.get("issues", []) for p in parsed]
        temp_df["requested_actions_raw"] = [p.get("requested_actions", []) for p in parsed]
        temp_df.to_parquet(out_dir / f"stage0_extract_{batch_idx:04d}.parquet", index=False)
        batches.append(temp_df)

    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()


def collect_problem_phrases(
    stage0_df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> list[str]:
    """Собирает уникальные формулировки для этапа 0b.

    По умолчанию берутся обе колонки:
    - issues_raw
    - requested_actions_raw

    Но можно передать конкретные колонки, чтобы строить таксономии раздельно.
    """

    if columns is None:
        columns = ("issues_raw", "requested_actions_raw")

    values: list[str] = []
    for column in columns:
        if column not in stage0_df.columns:
            continue
        for items in stage0_df[column]:
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                candidate = item.get("type") or item.get("description")
                if candidate:
                    values.append(str(candidate).strip())
    return sorted(set(v for v in values if v))


def format_problems_for_prompt(problems: Sequence[str]) -> str:
    """Форматирует проблемы в нумерованный список для prompt этапа 0b."""
    return "\n".join(f"{ix + 1}. {problem}" for ix, problem in enumerate(problems))


def run_stage0_taxonomy(
    chain: Any,
    problems: Sequence[str],
    output_path: str | Path,
    retries: int = 4,
    base_sleep_seconds: float = 2.0,
) -> str:
    """Запускает этап 0b и сохраняет JSON-таксономию."""
    prompt_input = {"problems": format_problems_for_prompt(problems)}
    result = invoke_with_backoff(
        chain,
        prompt_input,
        retries=retries,
        base_sleep_seconds=base_sleep_seconds,
    )
    if isinstance(result, TaxonomyResult):
        items = [item.model_dump() for item in result.items]
        content = json.dumps(items, ensure_ascii=False, indent=2)
    elif hasattr(result, "content"):
        content = result.content
    else:
        content = str(result)
    output_path = _ensure_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return content


# =========================
# Этап 1: классификация
# =========================


def run_stage1_classification(
    source_df: pd.DataFrame,
    chain: Any,
    output_dir: str | Path,
    text_column: str = "text",
    batch_size: int = 100,
    max_concurrency: int = 5,
    retries: int = 4,
    base_sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    """Запускает классификацию по справочнику и сохраняет чанки parquet.

    Выходные колонки:
    - issues_pred
    - requested_actions_pred
    """

    out_dir = _ensure_dir(output_dir)
    batches: list[pd.DataFrame] = []
    for batch_idx, temp_df in _iter_batches(source_df, batch_size):
        payloads = [{"text": str(text)} for text in temp_df[text_column].fillna("")]
        answers = batch_with_backoff(
            chain,
            payloads,
            max_concurrency=max_concurrency,
            retries=retries,
            base_sleep_seconds=base_sleep_seconds,
        )
        parsed = [_to_plain_dict(a, ClassificationAnswer()) for a in answers]

        temp_df["issues_pred"] = [p.get("issues", []) for p in parsed]
        temp_df["requested_actions_pred"] = [p.get("requested_actions", []) for p in parsed]
        temp_df.to_parquet(out_dir / f"stage1_classification_{batch_idx:04d}.parquet", index=False)
        batches.append(temp_df)

    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()


def explode_predictions(
    classified_df: pd.DataFrame,
    labels_column: str,
    text_column: str = "text",
) -> pd.DataFrame:
    """Преобразует list-колонку с метками в "плоскую" таблицу (по 1 метке в строке)."""
    rows: list[dict[str, Any]] = []
    for row_idx, row in classified_df.reset_index(drop=True).iterrows():
        labels = row.get(labels_column, [])
        if not isinstance(labels, list):
            continue
        for item in labels:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "row_id": row_idx,
                    "text": row.get(text_column, ""),
                    "category": item.get("category", ""),
                    "sub_category": item.get("sub_category", ""),
                    "description": item.get("description", ""),
                    "product_id": row.get("product_id", ""),
                    "product_category": row.get("product_category", ""),
                }
            )
    return pd.DataFrame(rows)


# =========================
# Этап 2: judge
# =========================


def run_judge(
    labels_df: pd.DataFrame,
    chain: Any,
    output_dir: str | Path,
    batch_size: int = 100,
    max_concurrency: int = 5,
    retries: int = 4,
    base_sleep_seconds: float = 2.0,
    file_prefix: str = "judge",
) -> pd.DataFrame:
    """Проверяет разметку (category/sub_category) через judge-цепочку."""
    out_dir = _ensure_dir(output_dir)
    batches: list[pd.DataFrame] = []

    for batch_idx, temp_df in _iter_batches(labels_df, batch_size):
        payloads = [
            {
                "text": str(text),
                "category": str(category),
                "sub_category": str(sub_category),
            }
            for text, category, sub_category in zip(
                temp_df["text"].fillna(""),
                temp_df["category"].fillna(""),
                temp_df["sub_category"].fillna(""),
            )
        ]
        answers = batch_with_backoff(
            chain,
            payloads,
            max_concurrency=max_concurrency,
            retries=retries,
            base_sleep_seconds=base_sleep_seconds,
        )
        parsed = [_to_plain_dict(a, JudgeAnswer(category=False, sub_category=False)) for a in answers]

        temp_df["judge_category_ok"] = [bool(p.get("category", False)) for p in parsed]
        temp_df["judge_sub_category_ok"] = [bool(p.get("sub_category", False)) for p in parsed]
        temp_df.to_parquet(out_dir / f"{file_prefix}_{batch_idx:04d}.parquet", index=False)
        batches.append(temp_df)

    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()


def run_legacy_judge(
    labels_df: pd.DataFrame,
    chain: Any,
    output_dir: str | Path,
    label_column: str = "label",
    text_column: str = "text",
    batch_size: int = 100,
    max_concurrency: int = 5,
    retries: int = 4,
    base_sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    """Опциональная оценка для legacy-плоских меток (одно поле decision)."""
    out_dir = _ensure_dir(output_dir)
    batches: list[pd.DataFrame] = []

    for batch_idx, temp_df in _iter_batches(labels_df, batch_size):
        payloads = [
            {"text": str(text), "label": str(label)}
            for text, label in zip(temp_df[text_column].fillna(""), temp_df[label_column].fillna(""))
        ]
        answers = batch_with_backoff(
            chain,
            payloads,
            max_concurrency=max_concurrency,
            retries=retries,
            base_sleep_seconds=base_sleep_seconds,
        )
        parsed = [_to_plain_dict(a, LegacyJudgeAnswer(decision=False)) for a in answers]
        temp_df["judge_decision"] = [bool(p.get("decision", False)) for p in parsed]
        temp_df.to_parquet(out_dir / f"judge_legacy_{batch_idx:04d}.parquet", index=False)
        batches.append(temp_df)

    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()


def read_parquet_chunks(path: str | Path, pattern: str = "*.parquet") -> pd.DataFrame:
    """Склеивает parquet-чанки из директории в один DataFrame."""
    directory = _ensure_path(path)
    files = sorted(directory.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def compute_judge_metrics(
    judge_df: pd.DataFrame,
    group_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Считает долю корректных category/sub_category.

    Если group_columns не заданы:
    - возвращается только одна строка "all".
    Если заданы:
    - добавляются групповые строки по этим колонкам.
    """

    if judge_df.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "judge_category_ok_rate",
                "judge_sub_category_ok_rate",
                "rows",
            ]
        )

    global_metrics = pd.DataFrame(
        [
            {
                "scope": "all",
                "judge_category_ok_rate": float(judge_df["judge_category_ok"].mean()),
                "judge_sub_category_ok_rate": float(judge_df["judge_sub_category_ok"].mean()),
                "rows": int(len(judge_df)),
            }
        ]
    )

    if not group_columns:
        return global_metrics

    grouped = (
        judge_df.groupby(list(group_columns), dropna=False)
        .agg(
            judge_category_ok_rate=("judge_category_ok", "mean"),
            judge_sub_category_ok_rate=("judge_sub_category_ok", "mean"),
            rows=("judge_category_ok", "size"),
        )
        .reset_index()
    )
    grouped["scope"] = grouped[list(group_columns)].astype(str).agg("|".join, axis=1)
    return pd.concat([global_metrics, grouped], ignore_index=True)
