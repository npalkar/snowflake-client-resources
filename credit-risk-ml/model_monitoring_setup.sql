-- ============================================================
-- MODEL MONITORING SETUP
--
-- Every statement here was executed against a live Snowflake
-- account and verified to return real metric values.
--
-- Prerequisites:
--   1. CREDIT_DEFAULT_MODEL registered (run the notebook)
--   2. CREDIT_RISK.ML.LOAN_APPLICATIONS exists
--      (your own table, or optional_sample_data.sql)
-- ============================================================

-- The model name in CREATE MODEL MONITOR resolves against the current
-- schema, so set context before creating the monitor.
USE SCHEMA CREDIT_RISK.ML;


-- ============================================================
-- STEP 1: Build the scoring table the monitor reads from
--
-- Note the two prediction columns. PROB_DEFAULT is the model's
-- probability; PREDICTED_DEFAULT is the approve/decline decision.
-- You need BOTH -- see the note in STEP 3.
-- ============================================================

CREATE OR REPLACE TABLE CREDIT_RISK.ML.SCORED_APPLICATIONS AS
WITH scored AS (
    SELECT
        APPLICANT_ID,
        -- Feature columns. The monitor discovers these automatically
        -- and computes drift on each one; you don't declare them.
        FICO_SCORE, VANTAGE_SCORE, NUM_OPEN_TRADES, NUM_TRADES_EVER,
        REVOLVING_UTILIZATION_PCT, TOTAL_REVOLVING_BALANCE,
        NUM_DELINQ_30_12M, NUM_DELINQ_60_12M, NUM_DELINQ_90_EVER,
        MONTHS_SINCE_OLDEST_TRADE, MONTHS_SINCE_NEWEST_TRADE, NUM_INQUIRIES_6M,
        BANKRUPTCY_FLAG, TOTAL_INSTALLMENT_BALANCE,
        PAYMENT_VELOCITY_TREND, BALANCE_TREND_12M, VENDOR_BANKRUPTCY_SCORE,
        -- Prediction score: the probability
        CREDIT_RISK.ML.CREDIT_DEFAULT_MODEL!PREDICT_PROBA(
            FICO_SCORE, VANTAGE_SCORE, NUM_OPEN_TRADES, NUM_TRADES_EVER,
            REVOLVING_UTILIZATION_PCT, TOTAL_REVOLVING_BALANCE,
            NUM_DELINQ_30_12M, NUM_DELINQ_60_12M, NUM_DELINQ_90_EVER,
            MONTHS_SINCE_OLDEST_TRADE, MONTHS_SINCE_NEWEST_TRADE, NUM_INQUIRIES_6M,
            BANKRUPTCY_FLAG, TOTAL_INSTALLMENT_BALANCE,
            PAYMENT_VELOCITY_TREND, BALANCE_TREND_12M, VENDOR_BANKRUPTCY_SCORE
        ):output_feature_1::FLOAT AS PROB_DEFAULT,
        -- Ground truth. Must be NUMBER for binary classification.
        DEFAULT_18M::NUMBER AS ACTUAL_DEFAULT,
        -- Must be TIMESTAMP_NTZ. Spread over 40 days here so there are
        -- multiple daily windows to aggregate; in production this is
        -- simply when the row was scored.
        DATEADD('day', -1 * UNIFORM(0, 39, RANDOM(42)), CURRENT_DATE())::TIMESTAMP_NTZ AS SCORED_AT
    FROM CREDIT_RISK.ML.LOAN_APPLICATIONS
)
SELECT
    *,
    -- Prediction class: the 0/1 decision at your policy threshold
    CASE WHEN PROB_DEFAULT > 0.35 THEN 1 ELSE 0 END::NUMBER AS PREDICTED_DEFAULT
FROM scored;


-- Validate before creating the monitor. Nulls, NaN, Inf, or probability
-- scores outside 0-1 cause refresh failure and eventual suspension.
SELECT
    COUNT(*)                                                            AS N,
    SUM(CASE WHEN PROB_DEFAULT IS NULL THEN 1 ELSE 0 END)               AS NULL_PROB,
    SUM(CASE WHEN PROB_DEFAULT < 0 OR PROB_DEFAULT > 1 THEN 1 ELSE 0 END) AS OUT_OF_RANGE,
    SUM(CASE WHEN ACTUAL_DEFAULT IS NULL THEN 1 ELSE 0 END)             AS NULL_ACTUAL,
    SUM(CASE WHEN SCORED_AT IS NULL THEN 1 ELSE 0 END)                  AS NULL_TS,
    COUNT(DISTINCT SCORED_AT::DATE)                                     AS DISTINCT_DAYS
FROM CREDIT_RISK.ML.SCORED_APPLICATIONS;
-- Expect all zeros except N and DISTINCT_DAYS.


-- ============================================================
-- STEP 2: Baseline snapshot
--
-- Drift is measured against this. Without a baseline the monitor
-- is created but drift metrics cannot be computed -- and adding one
-- later requires dropping and recreating the monitor.
--
-- In production, use a snapshot of the data the model was trained
-- on, or a period you consider representative of normal.
-- ============================================================

CREATE OR REPLACE TABLE CREDIT_RISK.ML.MONITOR_BASELINE AS
SELECT * EXCLUDE (SCORED_AT)
FROM CREDIT_RISK.ML.SCORED_APPLICATIONS
SAMPLE (30);


-- ============================================================
-- STEP 3: Create the monitor
--
-- Syntax notes, each of which produced a real failure in testing:
--
--   * Column parameters take QUOTED STRING arrays:
--       ID_COLUMNS = ( 'APPLICANT_ID' )      correct
--       ID_COLUMNS = ( APPLICANT_ID )        fails
--     But MODEL, SOURCE, BASELINE, TIMESTAMP_COLUMN and WAREHOUSE
--     are bare identifiers, unquoted.
--
--   * MODEL resolves against the CURRENT SCHEMA. Without the
--     USE SCHEMA above this fails with "MODEL does not exist or
--     not authorized" even though the model is right there.
--
--   * Declare BOTH prediction columns. With only a score column,
--     PRECISION / RECALL / F1_SCORE / CLASSIFICATION_ACCURACY
--     return NULL with no error raised.
-- ============================================================

CREATE OR REPLACE MODEL MONITOR CREDIT_DEFAULT_MONITOR WITH
    MODEL = CREDIT_DEFAULT_MODEL
    VERSION = 'V1'
    FUNCTION = 'PREDICT_PROBA'
    SOURCE = CREDIT_RISK.ML.SCORED_APPLICATIONS
    BASELINE = CREDIT_RISK.ML.MONITOR_BASELINE
    WAREHOUSE = COMPUTE_WH
    REFRESH_INTERVAL = '1 day'
    AGGREGATION_WINDOW = '1 day'
    TIMESTAMP_COLUMN = SCORED_AT
    ID_COLUMNS = ( 'APPLICANT_ID' )
    PREDICTION_SCORE_COLUMNS = ( 'PROB_DEFAULT' )
    PREDICTION_CLASS_COLUMNS = ( 'PREDICTED_DEFAULT' )
    ACTUAL_CLASS_COLUMNS = ( 'ACTUAL_DEFAULT' );


-- ============================================================
-- STEP 4: Confirm it is healthy
-- ============================================================

DESCRIBE MODEL MONITOR CREDIT_DEFAULT_MONITOR;
-- Check:
--   monitor_state                   = ACTIVE
--   aggregation_status              = ACTIVE for SOURCE_AGGREGATED
--                                     and ACCURACY_AGGREGATED
--   aggregation_last_error          = empty strings
--   columns -> numerical_columns    = the features it discovered


-- ============================================================
-- STEP 5: Performance metrics
--
-- ROC_AUC needs the prediction SCORE.
-- The other four need the prediction CLASS.
-- ============================================================

WITH m AS (
    SELECT 'ROC_AUC' AS METRIC, EVENT_TIMESTAMP, METRIC_VALUE, COUNT_USED
    FROM TABLE(MODEL_MONITOR_PERFORMANCE_METRIC('CREDIT_DEFAULT_MONITOR','ROC_AUC','1 DAY',
        DATEADD('DAY',-7,CURRENT_DATE()), CURRENT_DATE()))
    UNION ALL
    SELECT 'PRECISION', EVENT_TIMESTAMP, METRIC_VALUE, COUNT_USED
    FROM TABLE(MODEL_MONITOR_PERFORMANCE_METRIC('CREDIT_DEFAULT_MONITOR','PRECISION','1 DAY',
        DATEADD('DAY',-7,CURRENT_DATE()), CURRENT_DATE()))
    UNION ALL
    SELECT 'RECALL', EVENT_TIMESTAMP, METRIC_VALUE, COUNT_USED
    FROM TABLE(MODEL_MONITOR_PERFORMANCE_METRIC('CREDIT_DEFAULT_MONITOR','RECALL','1 DAY',
        DATEADD('DAY',-7,CURRENT_DATE()), CURRENT_DATE()))
    UNION ALL
    SELECT 'F1_SCORE', EVENT_TIMESTAMP, METRIC_VALUE, COUNT_USED
    FROM TABLE(MODEL_MONITOR_PERFORMANCE_METRIC('CREDIT_DEFAULT_MONITOR','F1_SCORE','1 DAY',
        DATEADD('DAY',-7,CURRENT_DATE()), CURRENT_DATE()))
    UNION ALL
    SELECT 'CLASSIFICATION_ACCURACY', EVENT_TIMESTAMP, METRIC_VALUE, COUNT_USED
    FROM TABLE(MODEL_MONITOR_PERFORMANCE_METRIC('CREDIT_DEFAULT_MONITOR','CLASSIFICATION_ACCURACY','1 DAY',
        DATEADD('DAY',-7,CURRENT_DATE()), CURRENT_DATE()))
)
SELECT
    EVENT_TIMESTAMP::DATE      AS DAY,
    METRIC,
    ROUND(METRIC_VALUE, 4)     AS VALUE,
    COUNT_USED                 AS ROWS_USED
FROM m
ORDER BY DAY DESC, METRIC;
-- All five should return values. Any NULLs in the four class-based
-- metrics mean PREDICTION_CLASS_COLUMNS was not declared.


-- ============================================================
-- STEP 6: Drift metrics
--
-- PSI on a FEATURE column is what is commonly called CSI.
-- PSI on the PREDICTION column is score drift.
-- ============================================================

SELECT
    EVENT_TIMESTAMP::DATE   AS DAY,
    COLUMN_NAME,
    ROUND(METRIC_VALUE, 5)  AS PSI
FROM TABLE(MODEL_MONITOR_DRIFT_METRIC(
    'CREDIT_DEFAULT_MONITOR', 'POPULATION_STABILITY_INDEX', 'FICO_SCORE', '1 DAY',
    DATEADD('DAY', -7, CURRENT_DATE()), CURRENT_DATE()
))
ORDER BY DAY DESC;


-- All four drift metrics, plus PSI on the prediction, for one day
WITH d AS (
    SELECT 'PSI on FICO_SCORE (CSI)' AS METRIC, METRIC_VALUE
    FROM TABLE(MODEL_MONITOR_DRIFT_METRIC('CREDIT_DEFAULT_MONITOR','POPULATION_STABILITY_INDEX',
        'FICO_SCORE','1 DAY', DATEADD('DAY',-1,CURRENT_DATE()), CURRENT_DATE()))
    UNION ALL
    SELECT 'PSI on PROB_DEFAULT', METRIC_VALUE
    FROM TABLE(MODEL_MONITOR_DRIFT_METRIC('CREDIT_DEFAULT_MONITOR','POPULATION_STABILITY_INDEX',
        'PROB_DEFAULT','1 DAY', DATEADD('DAY',-1,CURRENT_DATE()), CURRENT_DATE()))
    UNION ALL
    SELECT 'JENSEN_SHANNON', METRIC_VALUE
    FROM TABLE(MODEL_MONITOR_DRIFT_METRIC('CREDIT_DEFAULT_MONITOR','JENSEN_SHANNON',
        'FICO_SCORE','1 DAY', DATEADD('DAY',-1,CURRENT_DATE()), CURRENT_DATE()))
    UNION ALL
    SELECT 'WASSERSTEIN', METRIC_VALUE
    FROM TABLE(MODEL_MONITOR_DRIFT_METRIC('CREDIT_DEFAULT_MONITOR','WASSERSTEIN',
        'FICO_SCORE','1 DAY', DATEADD('DAY',-1,CURRENT_DATE()), CURRENT_DATE()))
    UNION ALL
    SELECT 'DIFFERENCE_OF_MEANS', METRIC_VALUE
    FROM TABLE(MODEL_MONITOR_DRIFT_METRIC('CREDIT_DEFAULT_MONITOR','DIFFERENCE_OF_MEANS',
        'FICO_SCORE','1 DAY', DATEADD('DAY',-1,CURRENT_DATE()), CURRENT_DATE()))
)
SELECT METRIC, ROUND(METRIC_VALUE, 5) AS VALUE FROM d;


-- ============================================================
-- STEP 7: Statistical metrics -- volume and data quality
-- ============================================================

WITH s AS (
    SELECT 'COUNT of predictions' AS METRIC, METRIC_VALUE
    FROM TABLE(MODEL_MONITOR_STAT_METRIC('CREDIT_DEFAULT_MONITOR','COUNT',
        'PROB_DEFAULT','1 DAY', DATEADD('DAY',-1,CURRENT_DATE()), CURRENT_DATE()))
    UNION ALL
    SELECT 'AVG predicted probability', METRIC_VALUE
    FROM TABLE(MODEL_MONITOR_STAT_METRIC('CREDIT_DEFAULT_MONITOR','AVG',
        'PROB_DEFAULT','1 DAY', DATEADD('DAY',-1,CURRENT_DATE()), CURRENT_DATE()))
    UNION ALL
    SELECT 'NULL count on FICO_SCORE', METRIC_VALUE
    FROM TABLE(MODEL_MONITOR_STAT_METRIC('CREDIT_DEFAULT_MONITOR','COUNT_NULL',
        'FICO_SCORE','1 DAY', DATEADD('DAY',-1,CURRENT_DATE()), CURRENT_DATE()))
)
SELECT METRIC, ROUND(METRIC_VALUE, 5) AS VALUE FROM s;


-- ============================================================
-- Housekeeping
-- ============================================================

-- Pause and resume
-- ALTER MODEL MONITOR CREDIT_DEFAULT_MONITOR SUSPEND;
-- ALTER MODEL MONITOR CREDIT_DEFAULT_MONITOR RESUME;

-- Tear down
-- DROP MODEL MONITOR IF EXISTS CREDIT_DEFAULT_MONITOR;
-- DROP TABLE IF EXISTS CREDIT_RISK.ML.MONITOR_BASELINE;
-- DROP TABLE IF EXISTS CREDIT_RISK.ML.SCORED_APPLICATIONS;
