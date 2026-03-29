-- Migration: Add date_key to dw.fact_churn
--
-- This aligns fact_churn with the same date-key pattern used by
-- fact_sales and fact_ab_test, allowing Power BI to relate
-- fact_churn[date_key] to dim_date[date_key] cleanly.
--
-- Note: rows where churn_date IS NULL will have date_key = NULL.
-- PostgreSQL allows NULL values in a foreign key column, so the
-- constraint below is still valid for those rows.

ALTER TABLE dw.fact_churn
    ADD COLUMN date_key INT;

UPDATE dw.fact_churn
SET date_key = TO_CHAR(churn_date, 'YYYYMMDD')::INT
WHERE churn_date IS NOT NULL;

-- Validation: ensure every non-NULL date_key exists in dim_date
-- before adding the foreign key.  If this returns rows, populate
-- dim_date for those dates first (e.g. by extending your dim_date
-- build procedure) and then re-run the ALTER TABLE below.
--
-- SELECT DISTINCT date_key
-- FROM   dw.fact_churn
-- WHERE  date_key IS NOT NULL
--   AND  date_key NOT IN (SELECT date_key FROM dw.dim_date);

ALTER TABLE dw.fact_churn
    ADD CONSTRAINT fk_fact_churn_date
    FOREIGN KEY (date_key) REFERENCES dw.dim_date(date_key);
