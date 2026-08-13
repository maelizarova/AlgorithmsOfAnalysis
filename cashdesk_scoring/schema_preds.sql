-- EMA_CASHDESK_PREDS
-- Одна строка = касса × score_date × forecast_date
-- Партиции: INTERVAL по score_date (день скоринга)

CREATE TABLE EMA_CASHDESK_PREDS (
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
    ON EMA_CASHDESK_PREDS (atdtmco_cashdesk_code, forecast_date)
    LOCAL;

-- Идемпотентный пересчёт дня:
--   DELETE FROM EMA_CASHDESK_PREDS WHERE score_date = :score_date;
--   затем APPEND строк за этот score_date.
--
-- One-time backfill перед первым prod-запуском:
--   залить историю >= error_window_days + data_lag_days
--   с полями central, ns_pred, quantile_used, is_closed.
