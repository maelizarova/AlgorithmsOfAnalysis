from __future__ import annotations

import glob
import json
from datetime import date, timedelta
from pathlib import Path

from prefect import flow, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

from tasks.scoring_functions import (
    clear_report_date,
    reading_data,
    scoring,
    to_sql,
)
from utils.result_builder import compute_report_window


MAX_WORKERS = 4
RETRIES = 1
RETRY_DELAY_SECONDS = 60


@flow(
    name="complaints_classification_daily",
    retries=RETRIES,
    retry_delay_seconds=RETRY_DELAY_SECONDS,
    task_runner=ConcurrentTaskRunner(max_workers=MAX_WORKERS),
)
def run_all_classifiers(
    report_date: str | None = None,
    lookback_days: int = 1,
    config_glob: str = "configs/*.json",
) -> list[int]:
    resolved_report_date = report_date or (date.today() - timedelta(days=1)).isoformat()
    start_date, end_date = compute_report_window(resolved_report_date, lookback_days=lookback_days)
    config_paths = sorted(glob.glob(config_glob))
    logger = get_run_logger()
    logger.info(
        "Запускаю классификаторы претензий: %s. Окно данных: [%s, %s)",
        len(config_paths),
        start_date,
        end_date,
    )

    if not config_paths:
        logger.warning("Не найдены конфиги по шаблону %s", config_glob)
        return []

    settings_path = Path(config_paths[0]).resolve().parent.parent / "settings.json"
    deployment_dir = str(settings_path.parent)
    with open(settings_path, encoding="utf-8") as f:
        settings = json.load(f)

    # clear day -> one Oracle extract -> per classifier: filter+score -> to_sql
    cleared = clear_report_date.submit(settings, resolved_report_date)
    source_future = reading_data.submit(
        settings,
        start_date,
        end_date,
        deployment_dir,
        wait_for=[cleared],
    )

    write_futures = []
    for config_path in config_paths:
        classifier_name = _peek_classifier_name(config_path)
        score_future = scoring.submit(
            source_future,
            config_path,
            resolved_report_date,
            classifier_name,
        )
        write_futures.append(
            to_sql.submit(
                score_future,
                classifier_name,
            )
        )

    return [future.result() for future in write_futures]


def _peek_classifier_name(config_path: str) -> str:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)["classifier_name"]


def _merge_settings(settings: dict, classifier_config: dict) -> dict:
    """Used by check_flow_debug.ipynb."""
    config = {**settings, **classifier_config}
    for section in ("llm", "runtime"):
        config[section] = {
            **settings.get(section, {}),
            **classifier_config.get(section, {}),
        }
    return config


if __name__ == "__main__":
    run_all_classifiers()
