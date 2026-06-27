from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from prefect import get_run_logger
from prefect.blocks.system import Secret
from toolbox import oracle

from deployment.llm_pipeline import (
    LLMConfig,
    build_classification_model,
    build_judge_issues_chain,
    build_judge_requests_chain,
    build_llm_from_config,
    build_stage1_chain,
    run_judge,
    run_stage1_classification,
)
from deployment.utils.result_builder import build_result_frame, filter_source_data
from deployment.utils.sql_writer import ClassificationSQLWriter


def create_oracle_engine(config: dict[str, Any]):
    password = Secret.load(config.get("oracle_password_secret_block", "pass-space")).get()
    return oracle.create_engine_space(config.get("oracle_username", "analytics"), password)


def read_source_data(engine, query_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    query_template = query_path.read_text(encoding="utf-8")
    query = query_template.format(start_date=start_date, end_date=end_date)
    return pd.read_sql(query, con=engine)


def run_classifier_scoring(
    config: dict[str, Any],
    *,
    score_date: str,
    start_date: str,
    end_date: str,
) -> int:
    logger = get_run_logger()
    classifier_name = config["classifier_name"]
    logger.info(
        "Running classifier %s for source window [%s, %s)",
        classifier_name,
        start_date,
        end_date,
    )

    engine = create_oracle_engine(config)
    query_path = _resolve_config_path(config, "query_path", "../queries/get_data.sql")
    source_df = read_source_data(engine, query_path, start_date, end_date)
    source_df = filter_source_data(
        source_df,
        product=config["product"],
        theme=config.get("theme"),
        category=config.get("category"),
    )

    if source_df.empty:
        logger.warning("No source rows for classifier %s", classifier_name)
        ClassificationSQLWriter(
            pd.DataFrame(),
            config.get("output_table", "ema_complaints_classification"),
            classifier_name,
            score_date,
        ).write_data(engine)
        return 0

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

    result_df = build_result_frame(
        classified_df,
        judge_issues_df,
        judge_requests_df,
        score_date=score_date,
        classifier_name=classifier_name,
    )
    ClassificationSQLWriter(
        result_df,
        config.get("output_table", "ema_complaints_classification"),
        classifier_name,
        score_date,
    ).write_data(engine)
    logger.info("Wrote %s rows for classifier %s", len(result_df), classifier_name)
    return len(result_df)


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


