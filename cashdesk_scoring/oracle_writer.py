import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sqlalchemy import text
from toolbox import oracle

logger = logging.getLogger(__name__)

PREDS_COLUMNS: Sequence[str] = (
    "score_date",
    "report_date",
    "atdtmco_cashdesk_code",
    "atdtmco_cashdesk_name",
    "atdtmco_cashdesk_name_trn",
    "forecast_date",
    "atdtmco_ns_pred_central",
    "atdtmco_ns_pred",
    "atdtmco_saldo_turn_pred",
    "quantile_used",
    "is_closed",
)


class CashdeskPredsWriter:
    def __init__(
        self,
        df: pd.DataFrame,
        table_name: str,
        score_date: str,
        batch_size: int = 100_000,
    ):
        missing = [column for column in PREDS_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(
                "В прогнозе нет колонок: {}".format(", ".join(missing))
            )
        self.df = df[list(PREDS_COLUMNS)].copy()
        self.table_name = table_name
        self.score_date = pd.Timestamp(score_date).strftime("%Y-%m-%d")
        self.batch_size = batch_size

    def _validate(self) -> None:
        key_columns = [
            "score_date",
            "atdtmco_cashdesk_code",
            "forecast_date",
        ]
        if self.df.duplicated(key_columns).any():
            raise ValueError("Перед записью в Oracle найдены дубли ключей")

        numeric_columns = [
            "atdtmco_ns_pred_central",
            "atdtmco_ns_pred",
            "atdtmco_saldo_turn_pred",
            "quantile_used",
        ]
        if not np.isfinite(
            self.df[numeric_columns].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Перед записью в Oracle найдены пропуски или невалидные числа"
            )
        self.df[numeric_columns] = self.df[numeric_columns].round(6)

    def _delete_score_date(self, engine) -> int:
        delete_query = text(
            """
            delete from {table_name}
            where score_date = to_date(:score_date, 'YYYY-MM-DD')
            """.format(table_name=self.table_name)
        )
        with engine.begin() as conn:
            result = conn.execute(
                delete_query,
                {"score_date": self.score_date},
            )
            deleted = int(result.rowcount or 0)
        logger.info(
            "Deleted %s rows for score_date=%s from %s",
            deleted,
            self.score_date,
            self.table_name,
        )
        return deleted

    def write_data(self, engine) -> int:
        self._validate()
        self._delete_score_date(engine)
        if self.df.empty:
            return 0

        oracle.write(
            self.df,
            engine,
            self.table_name,
            batch_size=self.batch_size,
            if_exists="append",
        )
        logger.info(
            "Inserted %s rows into %s for score_date=%s",
            len(self.df),
            self.table_name,
            self.score_date,
        )
        return len(self.df)
