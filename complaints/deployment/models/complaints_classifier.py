from __future__ import annotations

from typing import Any

from tasks.scoring_functions import run_classifier_scoring


class ComplaintsClassifier:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def run(self, *, report_date: str, start_date: str, end_date: str) -> int:
        return run_classifier_scoring(
            self.config,
            report_date=report_date,
            start_date=start_date,
            end_date=end_date,
        )
