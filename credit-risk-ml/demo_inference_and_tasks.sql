-- ============================================================
-- CREDIT DEFAULT MODEL DEMO - SQL Inference & Task Automation
-- Run this in Snowsight after the notebook training is complete
-- ============================================================

USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;
USE SCHEMA CREDIT_RISK.ML;

-- ============================================================
-- SECTION 1: Ad-Hoc SQL Inference
-- "Anyone with SQL can call the model"
-- ============================================================

SELECT 
    APPLICANT_ID,
    FICO_SCORE,
    REVOLVING_UTILIZATION_PCT,
    NUM_DELINQ_30_12M,
    ROUND(
        CREDIT_RISK.ML.CREDIT_DEFAULT_MODEL!PREDICT_PROBA(
            FICO_SCORE, VANTAGE_SCORE, NUM_OPEN_TRADES, NUM_TRADES_EVER,
            REVOLVING_UTILIZATION_PCT, TOTAL_REVOLVING_BALANCE,
            NUM_DELINQ_30_12M, NUM_DELINQ_60_12M, NUM_DELINQ_90_EVER,
            MONTHS_SINCE_OLDEST_TRADE, MONTHS_SINCE_NEWEST_TRADE, NUM_INQUIRIES_6M,
            BANKRUPTCY_FLAG, TOTAL_INSTALLMENT_BALANCE,
            PAYMENT_VELOCITY_TREND, BALANCE_TREND_12M, VENDOR_BANKRUPTCY_SCORE
        ):output_feature_1::FLOAT, 
    3) AS PROB_DEFAULT
FROM CREDIT_RISK.ML.LOAN_APPLICATIONS
LIMIT 20;


-- ============================================================
-- SECTION 2: Approve/Decline Business Logic
-- "The model gives probability, policy decides the threshold"
-- ============================================================

WITH scored AS (
    SELECT 
        APPLICANT_ID,
        FICO_SCORE,
        REVOLVING_UTILIZATION_PCT,
        CREDIT_RISK.ML.CREDIT_DEFAULT_MODEL!PREDICT_PROBA(
            FICO_SCORE, VANTAGE_SCORE, NUM_OPEN_TRADES, NUM_TRADES_EVER,
            REVOLVING_UTILIZATION_PCT, TOTAL_REVOLVING_BALANCE,
            NUM_DELINQ_30_12M, NUM_DELINQ_60_12M, NUM_DELINQ_90_EVER,
            MONTHS_SINCE_OLDEST_TRADE, MONTHS_SINCE_NEWEST_TRADE, NUM_INQUIRIES_6M,
            BANKRUPTCY_FLAG, TOTAL_INSTALLMENT_BALANCE,
            PAYMENT_VELOCITY_TREND, BALANCE_TREND_12M, VENDOR_BANKRUPTCY_SCORE
        ):output_feature_1::FLOAT AS PROB_DEFAULT,
        DEFAULT_18M AS ACTUAL_DEFAULT
    FROM CREDIT_RISK.ML.LOAN_APPLICATIONS
)
SELECT 
    APPLICANT_ID,
    FICO_SCORE,
    ROUND(PROB_DEFAULT, 3) AS PROB_DEFAULT,
    CASE WHEN PROB_DEFAULT > 0.35 THEN 'DECLINE' ELSE 'APPROVE' END AS DECISION,
    CASE WHEN ACTUAL_DEFAULT = 1 THEN 'Defaulted' ELSE 'No Default' END AS ACTUAL_OUTCOME
FROM scored
LIMIT 20;


-- ============================================================
-- SECTION 3: Results Table for Automated Scoring
-- ============================================================

CREATE TABLE IF NOT EXISTS CREDIT_RISK.ML.SCORED_APPLICATIONS (
    APPLICANT_ID VARCHAR,
    PROB_DEFAULT FLOAT,
    DECISION VARCHAR,
    SCORED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);


-- ============================================================
-- SECTION 4: Scheduled Task - Weekly Batch Scoring
-- "Replace manual monthly/weekly execution with automation"
-- ============================================================

CREATE OR REPLACE TASK CREDIT_RISK.ML.SCORE_NEW_APPLICATIONS
    WAREHOUSE = COMPUTE_WH
    SCHEDULE = 'USING CRON 0 6 * * MON America/New_York'
    COMMENT = 'Weekly credit scoring - scores all new applicants not yet in scored table'
AS
    INSERT INTO CREDIT_RISK.ML.SCORED_APPLICATIONS (APPLICANT_ID, PROB_DEFAULT, DECISION)
    SELECT 
        APPLICANT_ID,
        CREDIT_RISK.ML.CREDIT_DEFAULT_MODEL!PREDICT_PROBA(
            FICO_SCORE, VANTAGE_SCORE, NUM_OPEN_TRADES, NUM_TRADES_EVER,
            REVOLVING_UTILIZATION_PCT, TOTAL_REVOLVING_BALANCE,
            NUM_DELINQ_30_12M, NUM_DELINQ_60_12M, NUM_DELINQ_90_EVER,
            MONTHS_SINCE_OLDEST_TRADE, MONTHS_SINCE_NEWEST_TRADE, NUM_INQUIRIES_6M,
            BANKRUPTCY_FLAG, TOTAL_INSTALLMENT_BALANCE,
            PAYMENT_VELOCITY_TREND, BALANCE_TREND_12M, VENDOR_BANKRUPTCY_SCORE
        ):output_feature_1::FLOAT AS PROB_DEFAULT,
        CASE 
            WHEN PROB_DEFAULT > 0.35 THEN 'DECLINE' 
            ELSE 'APPROVE' 
        END AS DECISION
    FROM CREDIT_RISK.ML.LOAN_APPLICATIONS
    WHERE APPLICANT_ID NOT IN (SELECT APPLICANT_ID FROM CREDIT_RISK.ML.SCORED_APPLICATIONS);

-- Enable the task
ALTER TASK CREDIT_RISK.ML.SCORE_NEW_APPLICATIONS RESUME;


-- ============================================================
-- SECTION 5: Verify Task & Results
-- ============================================================

-- Show task is scheduled
SHOW TASKS IN SCHEMA CREDIT_RISK.ML;

-- Run it manually for the demo (don't wait for Monday)
EXECUTE TASK CREDIT_RISK.ML.SCORE_NEW_APPLICATIONS;

-- Check results
SELECT DECISION, COUNT(*) AS N, ROUND(AVG(PROB_DEFAULT), 3) AS AVG_PROB
FROM CREDIT_RISK.ML.SCORED_APPLICATIONS
GROUP BY DECISION;

-- Distribution of risk scores
SELECT 
    CASE 
        WHEN PROB_DEFAULT < 0.15 THEN '1: Very Low Risk'
        WHEN PROB_DEFAULT < 0.35 THEN '2: Low Risk'
        WHEN PROB_DEFAULT < 0.50 THEN '3: Medium Risk'
        WHEN PROB_DEFAULT < 0.70 THEN '4: High Risk'
        ELSE '5: Very High Risk'
    END AS RISK_TIER,
    COUNT(*) AS N,
    ROUND(AVG(PROB_DEFAULT), 3) AS AVG_PROB
FROM CREDIT_RISK.ML.SCORED_APPLICATIONS
GROUP BY 1
ORDER BY 1;
