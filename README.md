# Snowflake ML Quick Start Guide

A practical guide to building, deploying, and managing machine learning models entirely within Snowflake — from training to production inference with full governance.

## Why Snowflake for ML?

| Traditional (On-Prem VMs) | Snowflake |
|---------------------------|-----------|
| Fixed compute — limited hyperparameter tuning | Elastic warehouses scale in seconds, back to zero when idle |
| Data exports from SQL Server to Python environments | Data and training in one platform — no movement |
| Pickle files on shared drives, no versioning | Model Registry with versioning, lineage, and access controls |
| YAML files and conda envs for dependency management | Each model version is fully isolated with its own dependencies |
| Manual monthly/weekly scoring | Scheduled Tasks run automatically on a cron |
| No drift detection or monitoring | Native Model Monitor for drift and performance tracking |
| Pay for VMs 24/7 whether idle or not | Pay-per-second — idle costs nothing |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     SNOWFLAKE PLATFORM                        │
│                                                              │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────────┐  │
│  │   DATA   │───▶│ NOTEBOOK  │───▶│   MODEL REGISTRY     │  │
│  │ (Tables) │    │ (Training)│    │ (Versioned, Governed) │  │
│  └──────────┘    └───────────┘    └──────────┬───────────┘  │
│       │                                       │              │
│       │          ┌───────────┐               │              │
│       └─────────▶│ SCHEDULED │◀──────────────┘              │
│                  │   TASK    │                               │
│                  └─────┬─────┘                               │
│                        │                                     │
│                        ▼                                     │
│  ┌──────────────┐  ┌──────────┐  ┌────────────────────┐    │
│  │  STREAMLIT   │  │  SQL /   │  │  MODEL MONITOR     │    │
│  │  (End-User)  │  │  PowerBI │  │  (Drift/Perf)      │    │
│  └──────────────┘  └──────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Capabilities

### 1. Snowflake Notebooks — Python-Native Training

Train models using the same Python libraries you use today (XGBoost, scikit-learn, LightGBM, PyTorch) — running on elastic Snowflake compute instead of fixed VMs.

```python
from snowflake.snowpark.context import get_active_session
from xgboost import XGBClassifier

session = get_active_session()
df = session.table("MY_DATA").to_pandas()  # One line - no export

model = XGBClassifier(n_estimators=200, max_depth=5)
model.fit(X_train, y_train)
```

**Compute flexibility:**
- Medium warehouse (4 cores) for development
- 4XL warehouse (128 cores) for hyperparameter tuning — scales in 2 seconds
- Compute Pools for GPU workloads (deep learning, distributed training)
- Pay-per-second — auto-suspends when idle

---

### 2. Experiment Tracking — Compare and Reproduce

Record every training run's parameters, metrics, and artifacts. Compare runs side-by-side. No MLflow server required.

```python
from snowflake.ml.experiment import ExperimentTracking

exp = ExperimentTracking(session=session)
exp.set_experiment("CREDIT_DEFAULT_EXPERIMENT")

with exp.start_run("xgboost_v1"):
    exp.log_params({'max_depth': 5, 'learning_rate': 0.1})
    exp.log_metrics({'auc': 0.76, 'ks_statistic': 0.41})
```

**What you get:**
- Full parameter and metric history for every run
- Side-by-side comparison in Snowsight UI
- Audit trail: who ran what, when, with what results
- Direct path from "best experiment" to Model Registry

---

### 3. Model Registry — Governed, Versioned, Callable

Register trained models as first-class Snowflake objects. Each version is isolated with its own dependencies.

```python
from snowflake.ml.registry import Registry

reg = Registry(session=session, database_name="CREDIT_RISK", schema_name="ML")

mv = reg.log_model(
    model,
    model_name="CREDIT_DEFAULT_MODEL",
    version_name="v1",
    sample_input_data=X_test.head(10),
    conda_dependencies=["xgboost", "scikit-learn"],
    target_platforms=["WAREHOUSE"],
)
```

**What the Registry provides:**

| Capability | Benefit |
|-----------|---------|
| Versioning | v1, v2, v3 — roll back in one command |
| Dependency isolation | Each version carries its own packages — no conflicts |
| Input/output schema | Consumers know exactly what to pass and what they get |
| Lineage | Which table and columns trained this model |
| Explainability | Built-in SHAP values for any prediction |
| Access control | Same RBAC as all Snowflake objects |

---

### 4. SQL Inference — Anyone Can Call the Model

Once registered, the model is callable from SQL. No Python required for consumers.

```sql
SELECT 
    APPLICANT_ID,
    CREDIT_RISK.ML.CREDIT_DEFAULT_MODEL!PREDICT_PROBA(
        FICO_SCORE, VANTAGE_SCORE, NUM_OPEN_TRADES, ...
    ):output_feature_1::FLOAT AS PROB_DEFAULT,
    CASE WHEN PROB_DEFAULT > 0.35 THEN 'DECLINE' ELSE 'APPROVE' END AS DECISION
FROM NEW_APPLICATIONS;
```

**Who can call it:**
- Credit analysts via Snowsight
- BI teams via PowerBI/Tableau (same SQL connection they already use)
- Scheduled Tasks for batch automation
- Streamlit apps for end-user interfaces
- Any tool that speaks SQL

---

### 5. Scheduled Tasks — Automated Batch Scoring

Replace manual monthly/weekly model execution with automated tasks.

```sql
CREATE TASK SCORE_NEW_APPLICATIONS
    WAREHOUSE = COMPUTE_WH
    SCHEDULE = 'USING CRON 0 6 * * MON America/New_York'
AS
    INSERT INTO SCORED_APPLICATIONS (APPLICANT_ID, PROB_DEFAULT, DECISION)
    SELECT 
        APPLICANT_ID,
        MODEL!PREDICT_PROBA(...):output_feature_1::FLOAT,
        CASE WHEN PROB_DEFAULT > 0.35 THEN 'DECLINE' ELSE 'APPROVE' END
    FROM NEW_APPLICATIONS
    WHERE APPLICANT_ID NOT IN (SELECT APPLICANT_ID FROM SCORED_APPLICATIONS);
```

- Runs every Monday 6 AM (or any schedule)
- Alerts on failure
- No human intervention required
- Full execution history

---

### 6. Model Monitoring — Know Before It Breaks

Track feature drift, prediction drift, and model performance over time. Native to Snowflake — no additional tools.

```sql
CREATE MODEL MONITOR credit_default_monitor
  FOR MODEL CREDIT_RISK.ML.CREDIT_DEFAULT_MODEL VERSION v1
  SOURCE = CREDIT_RISK.ML.SCORED_APPLICATIONS
  PREDICTION_COLUMN = 'PROB_DEFAULT'
  ACTUAL_COLUMN = 'ACTUAL_DEFAULT'
  TIMESTAMP_COLUMN = 'SCORED_AT'
  WAREHOUSE = 'COMPUTE_WH';
```

**What it watches:**
- **Feature drift** — Are input distributions shifting?
- **Prediction drift** — Is the model approving/declining differently than historical baseline?
- **Performance metrics** — AUC, precision, recall recalculated as actual outcomes arrive
- **Data quality** — NULLs, out-of-range values, missing features

---

### 7. Streamlit — End-User Applications

Build web applications hosted in Snowflake for non-technical stakeholders. No separate infrastructure.

- Credit analysts get an approve/decline interface
- Risk managers get portfolio dashboards
- Executives get KPI views
- All powered by the same registered model via SQL

---

## Snowflake Marketplace — Data Enrichment

Access third-party data instantly to improve model performance:

- **Experian** — Premier Attributes, ClearView, vendor scores
- **Equifax** — Commercial credit data
- **TransUnion** — Consumer attributes
- **Alternative data** — Employment verification, bank transaction summaries, macroeconomic indicators

One click to add — appears as a live, always-current table. Test in an afternoon, buy if it works.

---

## Getting Started

1. **Migrate data to Snowflake** (in progress)
2. **Create a notebook** — same Python, elastic compute
3. **Train and register a model** — one `log_model()` call
4. **Call via SQL** — `MODEL!PREDICT_PROBA(...)` from anywhere
5. **Schedule scoring** — `CREATE TASK` with a cron
6. **Monitor** — `CREATE MODEL MONITOR` for drift detection

---

## Resources

- [Snowflake ML Documentation](https://docs.snowflake.com/en/developer-guide/snowflake-ml/overview)
- [Model Registry Guide](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/overview)
- [Snowflake Notebooks](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks)
- [Snowflake Tasks](https://docs.snowflake.com/en/user-guide/tasks-intro)
- [Snowflake Marketplace](https://app.snowflake.com/marketplace)
