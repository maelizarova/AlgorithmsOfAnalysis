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
    "report_date": types.Date(),
    "product": types.VARCHAR(500),
    "theme": types.VARCHAR(500),
    "category": types.VARCHAR(500),
    "classifier_name": types.VARCHAR(200),
    "type": types.VARCHAR(50),
    "class": types.VARCHAR(1000),
    "sub_class": types.VARCHAR(1000),
    "eval": types.Boolean(),
}


def delete_report_date_rows(engine, table_name: str, report_date: str) -> None:
    """Remove all classifier rows for one report_date before a full deployment re-run."""
    delete_query = text(
        f"""
        delete from {table_name}
        where report_date = to_date(:report_date, 'YYYY-MM-DD')
        """
    )
    with engine.begin() as conn:
        conn.execute(delete_query, {"report_date": report_date})


class ClassificationSQLWriter:
    def __init__(self, df: pd.DataFrame, table_name: str):
        self.df = df
        self.table_name = table_name

    def write_data(self, engine) -> None:
        if self.df.empty:
            return

        self.df.to_sql(
            name=self.table_name,
            con=engine,
            if_exists="append",
            index=False,
            dtype=CLASSIFICATION_DTYPE,
        )
