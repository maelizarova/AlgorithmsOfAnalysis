import os
from datetime import date

import pandas as pd
from prefect import flow, get_run_logger
from prefect.blocks.system import Secret
from toolbox import oracle

from cashdesk_forecast import load_settings, run_daily_score
from oracle_writer import CashdeskPredsWriter


SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")


def create_oracle_engine(settings: dict):
    password = Secret.load(settings["oracle_password_secret_block"]).get()
    return oracle.create_engine_cdw(settings["oracle_username"], password)


def write_predictions(
    engine,
    settings: dict,
    preds_df: pd.DataFrame,
    score_date: str,
) -> int:
    runtime = settings.get("runtime", {})
    writer = CashdeskPredsWriter(
        preds_df,
        table_name=settings["preds_table"],
        score_date=score_date,
        batch_size=int(runtime.get("write_batch_size", 100_000)),
    )
    return writer.write_data(engine)


@flow(
    name="ts_cashdesks_0826_v1",
    retries=1,
    retry_delay_seconds=60,
)
def ts_cashdesks_0826_v1(score_date=None):
    """
    Ежедневный скоринг потребности касс.

    Parameters
    ----------
    score_date:
        Дата скоринга YYYY-MM-DD. По умолчанию — сегодня.
    """
    logger = get_run_logger()
    settings = load_settings(str(SETTINGS_PATH))
    resolved_score_date = score_date or date.today().isoformat()
    logger.info("Cashdesk daily scoring for score_date=%s", resolved_score_date)

    engine = create_oracle_engine(settings)
    preds_df = run_daily_score(engine, settings, resolved_score_date)
    logger.info(
        "Built %s prediction rows; closed=%s; q=%s",
        len(preds_df),
        int(preds_df["is_closed"].sum()) if not preds_df.empty else 0,
        float(preds_df["quantile_used"].iloc[0]) if not preds_df.empty else None,
    )

    row_count = write_predictions(
        engine, settings, preds_df, resolved_score_date
    )
    logger.info("Wrote %s rows to %s", row_count, settings["preds_table"])
    return row_count


if __name__ == "__main__":
    ts_cashdesks_0826_v1()
