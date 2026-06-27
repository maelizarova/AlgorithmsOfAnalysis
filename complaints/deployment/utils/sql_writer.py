from __future__ import annotations

import sys

import oracledb
import pandas as pd
from sqlalchemy import text, types

oracledb.version = "8.3.0"
sys.modules["cx_Oracle"] = oracledb


CLASSIFICATION_DTYPE = {
    "claim_num": types.VARCHAR(100),
    "created": types.DateTime(),
    "score_date": types.Date(),
    "product": types.VARCHAR(500),
    "theme": types.VARCHAR(500),
    "category": types.VARCHAR(500),
    "classifier_name": types.VARCHAR(200),
    "type": types.VARCHAR(50),
    "class": types.VARCHAR(1000),
    "sub_class": types.VARCHAR(1000),
    "eval": types.Boolean(),
}


class ClassificationSQLWriter:
    def __init__(self, df: pd.DataFrame, table_name: str, classifier_name: str, score_date: str):
        self.df = df
        self.table_name = table_name
        self.classifier_name = classifier_name
        self.score_date = score_date

    def write_data(self, engine) -> None:
        delete_query = text(
            f"""
            delete from {self.table_name}
            where score_date = to_date(:score_date, 'YYYY-MM-DD')
              and classifier_name = :classifier_name
            """
        )
        with engine.begin() as conn:
            conn.execute(
                delete_query,
                {
                    "score_date": self.score_date,
                    "classifier_name": self.classifier_name,
                },
            )

        if self.df.empty:
            return

        self.df.to_sql(
            name=self.table_name,
            con=engine,
            if_exists="append",
            index=False,
            dtype=CLASSIFICATION_DTYPE,
        )
