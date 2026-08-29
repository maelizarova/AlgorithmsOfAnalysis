from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import get_run_logger, task
from prefect.blocks.system import Secret
from sqlalchemy import text
from toolbox import oracle

from llm_pipeline import (
    LLMConfig,
    build_classification_model,
    build_judge_issues_chain,
    build_judge_requests_chain,
    build_llm_from_config,
    build_stage1_chain,
    run_judge,
    run_stage1_classification,
)
from utils.result_builder import build_result_frame, filter_source_data
from utils.sql_writer import ClassificationSQLWriter, delete_report_date_rows


TASK_RETRIES = 1
TASK_RETRY_DELAY_SECONDS = 60


def create_oracle_engine(config: dict[str, Any]):
    password = Secret.load(config.get("oracle_password_secret_block", "pass-space")).get()
    return oracle.create_engine_space(config.get("oracle_username", "analytics"), password)


def read_source_data(engine, query_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    query_template = query_path.read_text(encoding="utf-8")
    query = query_template.format(start_date=start_date, end_date=end_date)
    with engine.connect() as conn:
        return pd.read_sql(text(query), con=conn)


def load_merged_config(config_path: str) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    settings_path = config_file.parent.parent / "settings.json"
    with open(settings_path, encoding="utf-8") as f:
        settings = json.load(f)
    with open(config_file, encoding="utf-8") as f:
        classifier_config = json.load(f)
    config = _merge_settings(settings, classifier_config)
    config["_config_dir"] = str(config_file.parent)
    return config


def _merge_settings(settings: dict, classifier_config: dict) -> dict:
    config = {**settings, **classifier_config}
    for section in ("llm", "runtime"):
        config[section] = {
            **settings.get(section, {}),
            **classifier_config.get(section, {}),
        }
    return config


def _read_source_window(
    settings: dict[str, Any],
    *,
    start_date: str,
    end_date: str,
    deployment_dir: str | Path,
) -> pd.DataFrame:
    engine = create_oracle_engine(settings)
    query_path = Path(settings.get("query_path", "queries/get_data.sql"))
    if not query_path.is_absolute():
        query_path = Path(deployment_dir) / query_path
    return read_source_data(engine, query_path.resolve(), start_date, end_date)


def _read_and_filter(
    config: dict[str, Any],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Single-classifier path: read SQL + filter (notebook / ComplaintsClassifier)."""
    engine = create_oracle_engine(config)
    query_path = _resolve_config_path(config, "query_path", "../queries/get_data.sql")
    source_df = read_source_data(engine, query_path, start_date, end_date)
    return filter_source_data(
        source_df,
        product=config["product"],
        theme=config.get("theme"),
        category=config.get("category"),
    )


def _score_source(
    config: dict[str, Any],
    source_df: pd.DataFrame,
    *,
    report_date: str,
    classifier_name: str,
) -> pd.DataFrame:
    if source_df.empty:
        return source_df.iloc[0:0].copy()

    text_column = config.get("text_column", "description_claim")
    if text_column not in source_df.columns:
        raise ValueError(f"Text column '{text_column}' is missing from source query result")

    llm = build_llm_from_config(_build_llm_config(config))
    product_context = _read_product_context(config)
    prompts_dir = _resolve_config_path(config, "prompts_dir", "../prompts")
    taxonomy_issues_path = _resolve_config_path(config, "taxonomy_issues_path")
    taxonomy_requests_path = _resolve_config_path(config, "taxonomy_requests_path")
    issues_table = taxonomy_issues_path.read_text(encoding="utf-8")
    requests_table = taxonomy_requests_path.read_text(encoding="utf-8")
    output_model = build_classification_model(taxonomy_issues_path, taxonomy_requests_path)

    classification_chain = build_stage1_chain(
        llm,
        prompts_dir / "stage1_classification.txt",
        issues_table,
        requests_table,
        product_context=product_context,
        output_model=output_model,
    )
    judge_issues_chain = build_judge_issues_chain(
        llm,
        prompts_dir / "judge_issues.txt",
        issues_table,
        product_context=product_context,
    )
    judge_requests_chain = build_judge_requests_chain(
        llm,
        prompts_dir / "judge_requests.txt",
        requests_table,
        product_context=product_context,
    )

    runtime = config.get("runtime", {})
    classified_df = run_stage1_classification(
        source_df,
        classification_chain,
        text_column=text_column,
        batch_size=runtime.get("batch_size", 100),
        max_concurrency=runtime.get("max_concurrency", 5),
        retries=runtime.get("retries", 4),
        base_sleep_seconds=runtime.get("backoff_base_seconds", 2.0),
    )
    judge_issues_df = run_judge(
        classified_df,
        judge_issues_chain,
        labels_column="issues_pred",
        text_column=text_column,
        batch_size=runtime.get("batch_size", 100),
        max_concurrency=runtime.get("max_concurrency", 5),
        retries=runtime.get("retries", 4),
        base_sleep_seconds=runtime.get("backoff_base_seconds", 2.0),
    )
    judge_requests_df = run_judge(
        classified_df,
        judge_requests_chain,
        labels_column="requested_actions_pred",
        text_column=text_column,
        batch_size=runtime.get("batch_size", 100),
        max_concurrency=runtime.get("max_concurrency", 5),
        retries=runtime.get("retries", 4),
        base_sleep_seconds=runtime.get("backoff_base_seconds", 2.0),
    )
    return build_result_frame(
        classified_df,
        judge_issues_df,
        judge_requests_df,
        report_date=report_date,
        classifier_name=classifier_name,
    )


def _write_result(config: dict[str, Any], result_df: pd.DataFrame) -> int:
    if result_df.empty:
        return 0
    engine = create_oracle_engine(config)
    output_table = config.get("output_table", "ema_complaints_classification")
    ClassificationSQLWriter(result_df, output_table).write_data(engine)
    return len(result_df)


@task(
    name="clear_report_date",
    retries=TASK_RETRIES,
    retry_delay_seconds=TASK_RETRY_DELAY_SECONDS,
)
def clear_report_date(settings: dict[str, Any], report_date: str) -> str:
    logger = get_run_logger()
    output_table = settings.get("output_table", "ema_complaints_classification")
    engine = create_oracle_engine(settings)
    logger.info("Удаляю строки за report_date=%s из %s", report_date, output_table)
    delete_report_date_rows(engine, output_table, report_date)
    return report_date


@task(
    name="reading_data",
    retries=TASK_RETRIES,
    retry_delay_seconds=TASK_RETRY_DELAY_SECONDS,
)
def reading_data(
    settings: dict[str, Any],
    start_date: str,
    end_date: str,
    deployment_dir: str,
) -> pd.DataFrame:
    """One shared Oracle extract for the day; classifiers filter later."""
    logger = get_run_logger()
    source_df = _read_source_window(
        settings,
        start_date=start_date,
        end_date=end_date,
        deployment_dir=deployment_dir,
    )
    logger.info(
        "Прочитано строк из Oracle: %s (окно [%s, %s))",
        len(source_df),
        start_date,
        end_date,
    )
    return source_df


@task(
    name="scoring",
    task_run_name="scoring[{classifier_name}]",
    retries=TASK_RETRIES,
    retry_delay_seconds=TASK_RETRY_DELAY_SECONDS,
)
def scoring(
    source_df: pd.DataFrame,
    config_path: str,
    report_date: str,
    classifier_name: str,
) -> dict[str, Any]:
    logger = get_run_logger()
    config = load_merged_config(config_path)
    filtered_df = filter_source_data(
        source_df,
        product=config["product"],
        theme=config.get("theme"),
        category=config.get("category"),
    )
    logger.info(
        "Фильтр %s: %s -> %s строк",
        classifier_name,
        len(source_df),
        len(filtered_df),
    )

    if filtered_df.empty:
        logger.warning("No source rows for classifier %s", classifier_name)
        empty = filtered_df.iloc[0:0].copy()
        return {
            "config": config,
            "result_df": empty,
            "classifier_name": classifier_name,
            "row_count": 0,
        }

    result_df = _score_source(
        config,
        filtered_df,
        report_date=report_date,
        classifier_name=classifier_name,
    )
    logger.info("Скоринг %s готов, строк результата: %s", classifier_name, len(result_df))
    return {
        "config": config,
        "result_df": result_df,
        "classifier_name": classifier_name,
        "row_count": len(result_df),
    }


@task(
    name="to_sql",
    task_run_name="to_sql[{classifier_name}]",
    retries=TASK_RETRIES,
    retry_delay_seconds=TASK_RETRY_DELAY_SECONDS,
)
def to_sql(score_payload: dict[str, Any], classifier_name: str) -> int:
    logger = get_run_logger()
    config = score_payload["config"]
    result_df = score_payload["result_df"]
    row_count = _write_result(config, result_df)
    logger.info("Wrote %s rows for classifier %s", row_count, classifier_name)
    return row_count


def run_classifier_scoring(
    config: dict[str, Any],
    *,
    report_date: str,
    start_date: str,
    end_date: str,
) -> int:
    """Synchronous path (notebook / ComplaintsClassifier), без Prefect tasks."""
    logger = get_run_logger()
    classifier_name = config["classifier_name"]
    logger.info(
        "Running classifier %s for source window [%s, %s)",
        classifier_name,
        start_date,
        end_date,
    )
    source_df = _read_and_filter(config, start_date, end_date)
    if source_df.empty:
        logger.warning("No source rows for classifier %s", classifier_name)
        return 0
    result_df = _score_source(
        config,
        source_df,
        report_date=report_date,
        classifier_name=classifier_name,
    )
    row_count = _write_result(config, result_df)
    logger.info("Wrote %s rows for classifier %s", row_count, classifier_name)
    return row_count


def _build_llm_config(config: dict[str, Any]) -> LLMConfig:
    llm_config = config["llm"]
    api_key = llm_config.get("api_key")
    if llm_config.get("api_key_secret_block"):
        api_key = Secret.load(llm_config["api_key_secret_block"]).get()
    if not api_key:
        raise ValueError("Set either llm.api_key or llm.api_key_secret_block in classifier config")

    return LLMConfig(
        api_key=api_key,
        base_url=llm_config["base_url"],
        model=llm_config["model"],
        temperature=llm_config.get("temperature", 0.0),
        timeout_seconds=llm_config.get("timeout_seconds", 60),
        model_kwargs=llm_config.get("model_kwargs", {}),
    )


def _read_product_context(config: dict[str, Any]) -> str:
    prompts_dir = _resolve_config_path(config, "prompts_dir", "../prompts")
    product = config["product"]
    candidates = []
    if config.get("theme"):
        candidates.append(prompts_dir / product / config["theme"] / "product_context.txt")
    if config.get("category"):
        candidates.append(prompts_dir / product / config["category"] / "product_context.txt")
    candidates.append(prompts_dir / product / "product_context.txt")

    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return ""


def _resolve_config_path(config: dict[str, Any], key: str, default: str | None = None) -> Path:
    value = config.get(key, default)
    if value is None:
        raise KeyError(f"Missing required config path: {key}")
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config.get("_config_dir", ".")) / path).resolve()
