-- ============================================================
-- OPTIONAL: Synthetic sample data
--
-- Creates a stand-in LOAN_APPLICATIONS table so you can run the
-- notebook end to end before your own data is available.
--
-- This is throwaway scaffolding. The column names are illustrative
-- stand-ins for bureau attributes, not a real Experian schema.
-- Once your data has landed, point the notebook at your table and
-- update the feature_cols list to match your actual columns.
-- ============================================================

CREATE DATABASE IF NOT EXISTS CREDIT_RISK;
CREATE SCHEMA IF NOT EXISTS CREDIT_RISK.ML;

USE SCHEMA CREDIT_RISK.ML;

-- 20,000 applicants with correlated bureau-style attributes and a
-- binary 18-month default flag.
CREATE OR REPLACE TABLE CREDIT_RISK.ML.LOAN_APPLICATIONS AS
WITH raw_data AS (
    SELECT
        'APP_' || LPAD(SEQ4()::VARCHAR, 6, '0') AS APPLICANT_ID,
        -- latent factors so the attributes correlate with each other
        NORMAL(0, 1, RANDOM()) AS z_credit,      -- overall credit quality
        NORMAL(0, 1, RANDOM()) AS z_behavior,    -- payment behavior
        NORMAL(0, 1, RANDOM()) AS z_seeking,     -- credit-seeking activity
        NORMAL(0, 1, RANDOM()) AS noise1,
        NORMAL(0, 1, RANDOM()) AS noise2,
        NORMAL(0, 1, RANDOM()) AS noise3,
        NORMAL(0, 1, RANDOM()) AS noise4,        -- irreducible uncertainty
        UNIFORM(0::FLOAT, 1::FLOAT, RANDOM()) AS u1,
        UNIFORM(0::FLOAT, 1::FLOAT, RANDOM()) AS u2
    FROM TABLE(GENERATOR(ROWCOUNT => 20000))
),
features AS (
    SELECT
        APPLICANT_ID,
        -- Vendor scores
        GREATEST(300, LEAST(850, ROUND(680 + 80 * z_credit + 15 * noise1)))::INT AS FICO_SCORE,
        GREATEST(300, LEAST(850, ROUND(670 + 75 * z_credit + 10 * (0.3 * noise1 + 0.7 * noise2))))::INT AS VANTAGE_SCORE,
        GREATEST(1, LEAST(999, ROUND(500 + 200 * z_credit + 50 * noise2)))::INT AS VENDOR_BANKRUPTCY_SCORE,
        -- Trade counts
        GREATEST(1, LEAST(30, ROUND(8 + 3 * z_credit + 2 * noise1)))::INT AS NUM_OPEN_TRADES,
        GREATEST(2, LEAST(60, ROUND(15 + 5 * z_credit + 4 * z_behavior + 3 * noise2)))::INT AS NUM_TRADES_EVER,
        -- Utilization and balances
        GREATEST(0, LEAST(100, ROUND(45 - 20 * z_credit + 10 * noise1 - 5 * z_behavior)))::FLOAT AS REVOLVING_UTILIZATION_PCT,
        GREATEST(0, ROUND(12000 - 5000 * z_credit + 8000 * u1 + 3000 * noise2))::FLOAT AS TOTAL_REVOLVING_BALANCE,
        GREATEST(0, ROUND(25000 + 10000 * noise1 - 5000 * z_credit + 15000 * u2))::FLOAT AS TOTAL_INSTALLMENT_BALANCE,
        -- Delinquency (zero-inflated: most applicants have none)
        GREATEST(0, LEAST(8, ROUND(-1.5 - 1.2 * z_credit - 0.8 * z_behavior + 1.5 * noise3)))::INT AS NUM_DELINQ_30_12M,
        GREATEST(0, LEAST(5, ROUND(-2.0 - 1.0 * z_credit - 0.6 * z_behavior + 1.2 * noise3)))::INT AS NUM_DELINQ_60_12M,
        GREATEST(0, LEAST(10, ROUND(-1.8 - 0.8 * z_credit - 0.5 * z_behavior + 1.8 * noise2)))::INT AS NUM_DELINQ_90_EVER,
        -- Credit history depth and recency
        GREATEST(6, LEAST(480, ROUND(180 + 60 * z_credit + 40 * noise1)))::INT AS MONTHS_SINCE_OLDEST_TRADE,
        GREATEST(0, LEAST(60, ROUND(12 - 5 * z_seeking + 4 * noise2)))::INT AS MONTHS_SINCE_NEWEST_TRADE,
        GREATEST(0, LEAST(12, ROUND(1.5 + 1.5 * z_seeking + 0.8 * noise3)))::INT AS NUM_INQUIRIES_6M,
        -- Public records
        CASE WHEN (-2.5 - 1.5 * z_credit - 0.5 * z_behavior + noise3) > 1.5 THEN 1 ELSE 0 END AS BANKRUPTCY_FLAG,
        -- Trended attributes (direction of change over time)
        GREATEST(-1, LEAST(1, ROUND(0.1 + 0.3 * z_behavior + 0.2 * z_credit + 0.15 * noise2, 2))) AS PAYMENT_VELOCITY_TREND,
        GREATEST(-1, LEAST(1, ROUND(-0.05 - 0.25 * z_credit + 0.3 * noise1 + 0.2 * z_seeking, 2))) AS BALANCE_TREND_12M,
        noise4, u1
    FROM raw_data
),
with_target AS (
    SELECT
        * EXCLUDE (noise4, u1),
        -- Default flag drawn from a logistic function of the attributes.
        -- The noise4 term is deliberately large: it represents causes of
        -- default that the attributes cannot explain (job loss, illness),
        -- which keeps model performance in a realistic range rather than
        -- letting the model separate the classes perfectly.
        CASE WHEN (
            1.0 / (1.0 + EXP(-(
                -0.3
                - 0.004  * (FICO_SCORE - 680)
                + 0.008  * REVOLVING_UTILIZATION_PCT
                + 0.15   * NUM_DELINQ_30_12M
                + 0.25   * NUM_DELINQ_60_12M
                + 0.10   * NUM_DELINQ_90_EVER
                - 0.002  * MONTHS_SINCE_OLDEST_TRADE
                + 0.05   * NUM_INQUIRIES_6M
                + 0.5    * BANKRUPTCY_FLAG
                - 0.3    * PAYMENT_VELOCITY_TREND
                + 0.2    * BALANCE_TREND_12M
                - 0.0005 * VENDOR_BANKRUPTCY_SCORE
                + 1.2    * noise4
            )))
        ) > u1 THEN 1 ELSE 0 END AS DEFAULT_18M
    FROM features
)
SELECT * FROM with_target;


-- ============================================================
-- Sanity checks
-- ============================================================

-- Overall shape. Expect ~20,000 rows and a default rate around 0.42.
SELECT
    COUNT(*)                          AS ROW_COUNT,
    ROUND(AVG(DEFAULT_18M), 3)        AS DEFAULT_RATE,
    ROUND(AVG(FICO_SCORE), 1)         AS AVG_FICO
FROM CREDIT_RISK.ML.LOAN_APPLICATIONS;

-- Default rate should decrease monotonically as score increases.
SELECT
    CASE
        WHEN FICO_SCORE < 580 THEN '1: <580'
        WHEN FICO_SCORE < 670 THEN '2: 580-669'
        WHEN FICO_SCORE < 740 THEN '3: 670-739'
        ELSE '4: 740+'
    END AS FICO_BAND,
    COUNT(*)                   AS N,
    ROUND(AVG(DEFAULT_18M), 3) AS DEFAULT_RATE
FROM CREDIT_RISK.ML.LOAN_APPLICATIONS
GROUP BY 1
ORDER BY 1;

-- Attributes should correlate the way real bureau data does:
-- the two scores track each other closely, and both utilization
-- and delinquency move opposite to score.
SELECT
    ROUND(CORR(FICO_SCORE, VANTAGE_SCORE), 3)              AS FICO_VS_VANTAGE,
    ROUND(CORR(FICO_SCORE, REVOLVING_UTILIZATION_PCT), 3)  AS FICO_VS_UTILIZATION,
    ROUND(CORR(FICO_SCORE, NUM_DELINQ_30_12M), 3)          AS FICO_VS_DELINQUENCY
FROM CREDIT_RISK.ML.LOAN_APPLICATIONS;
