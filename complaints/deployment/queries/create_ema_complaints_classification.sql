-- Output table for all classifiers (shared).
-- Partition by report_date (daily INTERVAL). No indexes.
--
-- report_date = last day of the report window (default: yesterday).
-- created = claim creation date from Siebel.
-- Full deployment re-run: DELETE WHERE report_date = :d at flow start,
-- then each classifier only appends.

CREATE TABLE ema_complaints_classification (
  claim_num         VARCHAR2(100),
  created           DATE,
  report_date       DATE NOT NULL,
  product           VARCHAR2(500),
  theme             VARCHAR2(500),
  category          VARCHAR2(500),
  classifier_name   VARCHAR2(200) NOT NULL,
  type              VARCHAR2(50),
  class             VARCHAR2(1000),
  sub_class         VARCHAR2(1000),
  eval              NUMBER(1)
)
PARTITION BY RANGE (report_date)
INTERVAL (NUMTODSINTERVAL(1, 'DAY'))
(
  PARTITION p_start VALUES LESS THAN (DATE '2026-01-01')
);
