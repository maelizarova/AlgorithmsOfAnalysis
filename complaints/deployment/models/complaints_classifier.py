from __future__ import annotations

from typing import Any

from deployment.tasks.scoring_functions import run_classifier_scoring


class ComplaintsClassifier:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def run(self, *, score_date: str, start_date: str, end_date: str) -> int:
        return run_classifier_scoring(
            self.config,
            score_date=score_date,
            start_date=start_date,
            end_date=end_date,
        )
