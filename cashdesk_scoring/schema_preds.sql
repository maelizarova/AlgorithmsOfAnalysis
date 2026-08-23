-- EMA_TS_CASHDESKS_0826_V1
-- Одна строка = касса × score_date × forecast_date
-- Партиции: INTERVAL по score_date (день скоринга)

CREATE TABLE EMA_TS_CASHDESKS_0826_V1 (
    score_date                 DATE           NOT NULL,
    report_date                DATE           NOT NULL,
    atdtmco_cashdesk_code      VARCHAR2(64)   NOT NULL,
    atdtmco_cashdesk_name      VARCHAR2(256),
    atdtmco_cashdesk_name_trn  VARCHAR2(512),
    forecast_date              DATE           NOT NULL,
    atdtmco_ns_pred_central    NUMBER,
    atdtmco_ns_pred            NUMBER,
    atdtmco_saldo_turn_pred    NUMBER,
    quantile_used              NUMBER,
    is_closed                  NUMBER(1)      NOT NULL,
    created_at                 TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT pk_ema_cashdesk_preds PRIMARY KEY (
        score_date,
        atdtmco_cashdesk_code,
        forecast_date
    )
)
PARTITION BY RANGE (score_date)
INTERVAL (NUMTODSINTERVAL(1, 'DAY'))
(
    PARTITION p_preds_start VALUES LESS THAN (DATE '2025-01-01')
);

CREATE INDEX ix_ema_cashdesk_preds_code_fd
    ON EMA_TS_CASHDESKS_0826_V1 (atdtmco_cashdesk_code, forecast_date)
    LOCAL;

-- Идемпотентный пересчёт дня:
--   DELETE FROM EMA_TS_CASHDESKS_0826_V1 WHERE score_date = :score_date;
--   затем APPEND строк за этот score_date.
--
-- One-time backfill перед первым prod-запуском:
--   ноутбук cashdesk_backfill_prod_logic.ipynb (параллельный ретро
--   с prod-логикой: NaN в обучении, zero только по расписанию,
--   окно 280 / shrinkage 10 / q=0.13), затем ячейка
--   WRITE_BACKFILL_TO_ORACLE = True.
