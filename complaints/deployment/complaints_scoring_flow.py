from __future__ import annotations

import glob
import json
from datetime import date
from pathlib import Path

from prefect import Task, flow, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

from deployment.models.complaints_classifier import ComplaintsClassifier
from deployment.utils.result_builder import compute_score_window


MAX_WORKERS = 4
RETRIES = 1
RETRY_DELAY_SECONDS = 60


def run_single_classifier(config_path: str, score_date: str, start_date: str, end_date: str) -> int:
    logger = get_run_logger()
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    config["_config_dir"] = str(Path(config_path).parent.resolve())

    classifier_name = config["classifier_name"]
    logger.info("Running complaints classifier %s", classifier_name)
    classifier = ComplaintsClassifier(config)
    row_count = classifier.run(score_date=score_date, start_date=start_date, end_date=end_date)
    logger.info("Finished complaints classifier %s with %s rows", classifier_name, row_count)
    return row_count


@flow(
    name="complaints_classification_daily",
    retries=RETRIES,
    retry_delay_seconds=RETRY_DELAY_SECONDS,
    task_runner=ConcurrentTaskRunner(max_workers=MAX_WORKERS),
)
def run_all_classifiers(
    score_date: str | None = None,
    lookback_days: int = 1,
    config_glob: str = "deployment/configs/*.json",
) -> list[int]:
    resolved_score_date = score_date or date.today().isoformat()
    start_date, end_date = compute_score_window(resolved_score_date, lookback_days=lookback_days)
    config_paths = sorted(glob.glob(config_glob))
    logger = get_run_logger()
    logger.info(
        "Running %s complaint classifiers for source window [%s, %s)",
        len(config_paths),
        start_date,
        end_date,
    )

    task_runs = []
    for config_path in config_paths:
        config_name = Path(config_path).stem
        classifier_task = Task(
            fn=run_single_classifier,
            name=f"run_complaints_classifier_{config_name}",
            task_run_name=f"run_complaints_classifier[{config_name}]",
            retries=RETRIES,
            retry_delay_seconds=RETRY_DELAY_SECONDS,
        )
        task_runs.append(classifier_task.submit(config_path, resolved_score_date, start_date, end_date))

    return [task_run.result() for task_run in task_runs]


if __name__ == "__main__":
    run_all_classifiers()
