![Type](https://img.shields.io/badge/Type-Guide-blue)
![Deploy](https://img.shields.io/badge/Deploy-None-lightgrey)
![Review](https://img.shields.io/badge/Review-2026--12--31-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

# Machine Learning on Snowflake: One Platform, End to End

Most ML teams stitch together a database, a compute environment, an experiment tracker, a model store, a scheduler, and a monitoring vendor. This guide shows the same lifecycle running entirely inside Snowflake — same Python, same libraries, without the plumbing between them.

> **Just want to train and register a model?** Jump to [Notebooks](#1-snowflake-notebooks--python-native-training) and [Model Registry](#4-model-registry--governed-versioned-callable). The rest fills in around it.

**Audience:** Data scientists and ML engineers moving workloads onto Snowflake
**Companion code:** [`credit_default_model.ipynb`](credit_default_model.ipynb) · [`demo_inference_and_tasks.sql`](demo_inference_and_tasks.sql) · [`streamlit_app.py`](streamlit_app.py)

> **Reference only — no support provided.** Validate against the linked documentation before production use. Snowflake ML features move quickly; a few noted below are in preview and may change.

---

## Start Here: What Are You Trying to Do?

| If you want to... | Go to |
|---|---|
| Train a model on Snowflake compute instead of a VM | [1. Notebooks](#1-snowflake-notebooks--python-native-training) |
| Stop duplicating feature logic between training and scoring | [2. Feature Store](#2-feature-store--managed-feature-pipelines) |
| Compare many experiments without losing track of them | [3. Experiment Tracking](#3-experiment-tracking--compare-and-reproduce) |
| Version models and stop managing conda environments | [4. Model Registry](#4-model-registry--governed-versioned-callable) |
| Ship preprocessing along with the model | [5. Preprocessing Pipelines](#5-preprocessing-pipelines-in-the-registry) |
| Let analysts and BI tools call the model | [6. SQL Inference](#6-sql-inference--anyone-can-call-the-model) |
| Replace manual monthly/weekly scoring runs | [7. Tasks & Orchestration](#7-scheduled-tasks--python-orchestration) |
| Handle upstream data that arrives late | [8. Data Dependencies](#8-handling-upstream-data-dependencies) |
| Track drift and performance without a monitoring vendor | [9. Model Monitoring](#9-model-monitoring--built-in-metrics) |
| Give non-technical stakeholders a UI | [10. Streamlit](#10-streamlit--end-user-applications) |
| Connect Snowflake to your Git repo | [11. GitHub](#11-connecting-snowflake-to-github) |

Sections run in workflow order, so reading top to bottom also works.

---

## Why Snowflake for ML?

| Traditional (On-Prem VMs) | Snowflake |
|---------------------------|-----------|
| Fixed compute — limited hyperparameter tuning | Elastic warehouses scale in seconds, back to zero when idle |
| Data exports from SQL Server to Python environments | Data and training in one platform — no movement |
| Feature logic duplicated between training and scoring scripts | Feature Store defines features once, serves both paths |
| Pickle files on shared drives, no versioning | Model Registry with versioning, lineage, and access controls |
| YAML files and conda envs for dependency management | Each model version is fully isolated with its own dependencies |
| Manual monthly/weekly scoring | Scheduled and event-triggered Tasks run automatically |
| No drift detection or monitoring | Native Model Monitor with built-in PSI, AUC, precision, recall |
| Pay for VMs 24/7 whether idle or not | Pay-per-second — idle costs nothing |

---

## Architecture Overview

```mermaid
flowchart TD
    Data[("Data Tables")] --> FS["Feature Store"]
    FS --> NB["Notebook Training"]
    NB --> Exp["Experiment Tracking"]
    Exp --> Reg["Model Registry"]

    Reg --> Task["Task: scheduled or triggered"]
    FS --> Task

    Task --> Scored[("Scored Results")]

    Scored --> SQL["SQL / PowerBI"]
    Scored --> ST["Streamlit App"]
    Scored --> Mon["Model Monitor"]

    Mon -.->|"retrain signal"| NB
```

Everything in that diagram is a Snowflake object governed by the same RBAC — no external services, no data egress.

---


## 1. Snowflake Notebooks — Python-Native Training

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
- 4XL warehouse (128 cores) for hyperparameter tuning — resizes in seconds
- Compute Pools for GPU workloads (deep learning, distributed training)
- Pay-per-second — auto-suspends when idle

```sql
-- Scale up for a tuning sweep, scale back down when done
ALTER WAREHOUSE COMPUTE_WH SET WAREHOUSE_SIZE = '4X-LARGE';
-- ... run experiments ...
ALTER WAREHOUSE COMPUTE_WH SET WAREHOUSE_SIZE = 'MEDIUM';
```

---

## 2. Feature Store — Managed Feature Pipelines

The Feature Store is the answer to "how do we manage our feature engineering pipelines?" Rather than maintaining transformation logic in Python scripts that run before training and again before scoring, you define each feature once and Snowflake keeps it fresh and serves it to both paths.

### Core concepts

| Concept | What it is |
|---------|-----------|
| **Entity** | The thing features describe, and its join key(s) — e.g. an applicant, keyed by `APPLICANT_ID` |
| **Feature View** | A set of related features plus the logic that computes them. Backed by a Dynamic Table when managed by Snowflake |
| **Training Set** | A point-in-time-correct dataset assembled from feature views for model training |

### Defining features

```python
from snowflake.ml.feature_store import FeatureStore, Entity, FeatureView, CreationMode

fs = FeatureStore(
    session=session,
    database="CREDIT_RISK",
    name="FEATURE_STORE",
    default_warehouse="COMPUTE_WH",
    creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
)

# Define the entity
applicant = Entity(name="APPLICANT", join_keys=["APPLICANT_ID"])
fs.register_entity(applicant)

# Define a feature view with your transformation logic
bureau_features = session.sql("""
    SELECT
        APPLICANT_ID,
        AS_OF_TS,
        FICO_SCORE,
        REVOLVING_UTILIZATION_PCT,
        NUM_DELINQ_30_12M,
        TOTAL_REVOLVING_BALANCE / NULLIF(TOTAL_CREDIT_LIMIT, 0) AS AGG_UTILIZATION,
        NUM_DELINQ_60_12M + NUM_DELINQ_90_EVER AS SEVERE_DELINQ_COUNT
    FROM RAW_BUREAU_PULLS
""")

fv = FeatureView(
    name="BUREAU_FEATURES",
    entities=[applicant],
    feature_df=bureau_features,
    timestamp_col="AS_OF_TS",
    refresh_freq="1 day",        # Snowflake keeps this fresh automatically
    desc="Experian Premier Attributes and derived ratios",
)

registered_fv = fs.register_feature_view(fv, version="v1")
```

Because `refresh_freq` is set, the Feature Store creates and maintains a Dynamic Table behind the scenes. The pipeline runs itself — no task to write, no orchestration to manage.

### Point-in-time correctness

This matters enormously for a credit model with an 18-month performance window. If you train on today's feature values but the label reflects behavior from 18 months ago, you leak future information and the model looks far better in development than it performs in production.

`generate_training_set` solves this with an ASOF join — for each training row it selects the feature values as they existed at that row's timestamp:

```python
# Spine = the applicants, their application timestamps, and the outcome
spine_df = session.sql("""
    SELECT APPLICANT_ID, APPLICATION_TS, DEFAULT_18M
    FROM HISTORICAL_APPLICATIONS
""")

training_set = fs.generate_training_set(
    spine_df=spine_df,
    features=[registered_fv],
    spine_timestamp_col="APPLICATION_TS",   # features as of application time
    spine_label_cols=["DEFAULT_18M"],
)

training_df = training_set.to_pandas()
```

For each applicant, this retrieves their bureau attributes as of the day they applied — not today's values. No manual window logic, no risk of leakage.

### Training and serving from the same definition

The same feature view feeds inference, which eliminates train/serve skew:

```python
prediction_df = fs.retrieve_feature_values(
    spine_df=new_applications_df,
    features=[registered_fv],
    spine_timestamp_col="APPLICATION_TS",
)
```

The features the model sees in production are computed by exactly the same logic that produced its training data.

### Why this matters for a multi-model team

| Without Feature Store | With Feature Store |
|----------------------|-------------------|
| Feature logic duplicated in training and scoring scripts — they drift apart | One definition serves both |
| Each model re-derives the same features from raw data | Define `SEVERE_DELINQ_COUNT` once; credit, fraud, and collections models all reuse it |
| Point-in-time correctness handled manually with window functions | ASOF join handled by `generate_training_set` |
| No catalog — "does someone already compute utilization?" | Discoverable registry of entities and feature views |
| Feature pipelines are hand-written tasks | Managed Dynamic Tables with a declared refresh frequency |

### Feature patterns available

| Pattern | Use when |
|---------|----------|
| **External** | Features already computed and maintained outside the Feature Store (static or slow-changing) |
| **Managed** | Snowflake computes and refreshes on a schedule — the common case |
| **Time-windowed** | Trailing aggregates like spend-7d or delinquencies-90d. Uses incremental tiling to keep cost down |
| **Append-only** | Retains a full history of feature snapshots for deeper point-in-time reconstruction |
| **Online** | Millisecond lookups for real-time inference (Postgres-backed) |
| **Rollup** | Aggregate lower-level features to a higher-level entity (card to account, product to category) |
| **Iceberg** | Features stored as Dynamic Iceberg Tables for cross-engine access |

### Time-windowed aggregations

Instead of writing rolling-window SQL by hand, declare the windows:

```python
from snowflake.ml.feature_store import Feature

features = [
    Feature.sum("PAYMENT_AMOUNT", "30d").alias("TOTAL_PAYMENTS_30D"),
    Feature.count("DELINQUENCY_EVENT", "90d").alias("DELINQ_EVENTS_90D"),
    Feature.avg("BALANCE", "7d").alias("AVG_BALANCE_7D"),
    # offset shifts the window back for period-over-period comparisons
    Feature.sum("PAYMENT_AMOUNT", "30d", offset="30d").alias("TOTAL_PAYMENTS_PRIOR_30D"),
]

fv = FeatureView(
    name="PAYMENT_BEHAVIOR",
    entities=[applicant],
    feature_df=session.table("PAYMENT_EVENTS"),
    timestamp_col="EVENT_TS",
    feature_granularity="1d",    # tile size
    refresh_freq="1d",
    features=features,
)
```

Snowflake maintains partial aggregates (tiles) incrementally rather than rescanning raw events, then stitches them at query time. Training sets that include tiled feature views need `join_method="cte"`:

```python
training_set = fs.generate_training_set(
    spine_df=spine_df,
    features=[registered_fv, registered_agg_fv],
    spine_timestamp_col="APPLICATION_TS",
    spine_label_cols=["DEFAULT_18M"],
    join_method="cte",
)
```

Requires `snowflake-ml-python` 1.24.0 or later for time-windowed aggregation; `generate_training_set` requires 1.5.4 or later.

### Notes worth knowing up front

- Training sets are ephemeral Snowpark DataFrames by default. Pass `save_as="<table>"` to materialize, or use `generate_dataset` for an immutable, versioned Snowflake Dataset when you need reproducibility guarantees.
- When reading offline feature views through the Python SDK, set `ALTER SESSION SET TIMEZONE = 'UTC'` first. Offline reads can interpret timestamps using the session timezone, which can produce values that disagree with online reads.
- If you build feature pipelines as Dynamic Tables outside the Feature Store, you can still register a view-based Feature View on top of them. Don't apply Feature Store object tags to those Dynamic Tables directly.

**Docs:** [Feature Store overview](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview) | [Advanced feature engineering](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/advanced-feature-engineering) | [Training and inference](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/modeling)

---

## 3. Experiment Tracking — Compare and Reproduce

Record every training run's parameters, metrics, and artifacts. Compare runs side-by-side. No MLflow server to install or maintain.

```python
from snowflake.ml.experiment import ExperimentTracking

exp = ExperimentTracking(session=session, database_name="CREDIT_RISK", schema_name="ML")
exp.set_experiment("CREDIT_DEFAULT_EXPERIMENT")

with exp.start_run("xgboost_v1"):
    exp.log_params({'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 200})
    exp.log_metrics({'auc': 0.76, 'ks_statistic': 0.41, 'gini': 0.52})
```

**What you get:**

- Full parameter and metric history for every run
- Side-by-side comparison in Snowsight
- Audit trail: who ran what, when, with what results
- A direct path from "best experiment" to Model Registry

Autologging callbacks are available for XGBoost, LightGBM, and Keras — metrics are captured per training iteration with no manual logging.

---

## 4. Model Registry — Governed, Versioned, Callable

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
    comment="XGBoost credit default model. AUC=0.76, KS=0.41",
)
```

**What the Registry provides:**

| Capability | Benefit |
|-----------|---------|
| Versioning | v1, v2, v3 all callable simultaneously — roll back by re-pointing an alias |
| Dependency isolation | Each version records its own package versions. One model on xgboost 1.7 and another on 2.0 coexist without conflict — no conda environments to activate, no YAML files to keep in sync |
| Input/output schema | Consumers know exactly what to pass and what they get back |
| Lineage | Which table and columns trained this version |
| Explainability | SHAP values enabled automatically when logged with `sample_input_data` and a warehouse target |
| Access control | Standard Snowflake RBAC |

Pass a **Snowpark** DataFrame (not pandas) as `sample_input_data` to capture data lineage.

---

## 5. Preprocessing Pipelines in the Registry

You can register a preprocessing pipeline together with the model so one call does both transformation and scoring.

### Option A: sklearn Pipeline (recommended)

The registry natively supports `sklearn.pipeline.Pipeline`, including pipelines that end in an XGBoost model.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(n_estimators=200, max_depth=5)),
])
pipe.fit(X_train, y_train)

mv = reg.log_model(
    pipe,
    model_name="CREDIT_DEFAULT_PIPELINE",
    version_name="v1",
    sample_input_data=X_test.head(10),
    conda_dependencies=["scikit-learn", "xgboost"],
    target_platforms=["WAREHOUSE"],
)
```

Raw input goes in, a score comes out — the transformation is baked in:

```sql
SELECT CREDIT_RISK.ML.CREDIT_DEFAULT_PIPELINE!PREDICT_PROBA(<raw columns>)
FROM NEW_APPLICATIONS;
```

### Option B: CustomModel for anything more complex

If your preprocessing isn't expressible as an sklearn Pipeline, or you need to chain several models:

```python
from snowflake.ml.model import custom_model
import pandas as pd

mc = custom_model.ModelContext(
    feature_preproc=my_preprocessor,   # sklearn pipeline object
    model=my_xgb_model,                # any natively supported model type
)

class CreditScoringModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        features = self.context['feature_preproc'].transform(input)
        probs = self.context['model'].predict_proba(features)
        return pd.DataFrame({'prob_default': probs[:, 1]})

mv = reg.log_model(
    CreditScoringModel(mc),
    model_name="CREDIT_DEFAULT_CUSTOM",
    version_name="v1",
    sample_input_data=X_test.head(10),
    conda_dependencies=["scikit-learn", "xgboost"],
)
```

Supported model objects passed into the context are serialized automatically — no manual pickling.

Two things to get right:

- Always access model objects through `self.context[...]` inside the class rather than assigning them to `self` in `__init__`. Direct assignment captures a second copy in a closure and significantly inflates the serialized model.
- Inference methods must handle **multi-row DataFrames**. Server-side batching can combine single-record requests from different callers into one DataFrame.

To bring helper modules along, use `code_paths=["src/preprocessing_utils"]`.

### Which to use

For feature engineering that's shared across models or needs to stay fresh, use the **Feature Store**. For row-level transformations that are intrinsic to one model — imputation, scaling, encoding — put them in the **registered pipeline** so they travel with the model.

**Docs:** [Pre-processing and post-processing with models](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/custom-processing-with-models)

---

## 6. SQL Inference — Anyone Can Call the Model

Once registered, the model is callable from SQL. No Python required for consumers.

```sql
WITH scored AS (
    SELECT
        APPLICANT_ID,
        CREDIT_RISK.ML.CREDIT_DEFAULT_MODEL!PREDICT_PROBA(
            FICO_SCORE, VANTAGE_SCORE, NUM_OPEN_TRADES, ...
        ):output_feature_1::FLOAT AS PROB_DEFAULT
    FROM NEW_APPLICATIONS
)
SELECT
    APPLICANT_ID,
    ROUND(PROB_DEFAULT, 3) AS PROB_DEFAULT,
    CASE WHEN PROB_DEFAULT > 0.35 THEN 'DECLINE' ELSE 'APPROVE' END AS DECISION
FROM scored;
```

`PREDICT_PROBA` returns class probabilities; `PREDICT` returns the hard class label. For risk scoring you almost always want `PREDICT_PROBA` — `output_feature_1` is the probability of the positive class.

**Who can call it:**

- Credit analysts via Snowsight
- BI teams via PowerBI or Tableau — the same SQL connection they already use
- Scheduled Tasks for batch automation
- Streamlit apps for end-user interfaces
- Any tool that speaks SQL

---

## 7. Scheduled Tasks & Python Orchestration

### SQL scoring on a schedule

```sql
CREATE OR REPLACE TASK SCORE_NEW_APPLICATIONS
    WAREHOUSE = COMPUTE_WH
    SCHEDULE = 'USING CRON 0 6 * * MON America/New_York'
AS
    INSERT INTO SCORED_APPLICATIONS (APPLICANT_ID, PROB_DEFAULT, DECISION)
    SELECT
        APPLICANT_ID,
        CREDIT_RISK.ML.CREDIT_DEFAULT_MODEL!PREDICT_PROBA(...):output_feature_1::FLOAT,
        CASE WHEN PROB_DEFAULT > 0.35 THEN 'DECLINE' ELSE 'APPROVE' END
    FROM NEW_APPLICATIONS
    WHERE APPLICANT_ID NOT IN (SELECT APPLICANT_ID FROM SCORED_APPLICATIONS);

ALTER TASK SCORE_NEW_APPLICATIONS RESUME;
```

### Multi-step Python scripts

Production flows are rarely one step — data cleaning, transformation, scoring, logging, notifications. You don't need to rewrite any of that as SQL. Wrap the Python in a stored procedure:

```python
def monthly_scoring(session):
    raw = session.table("RAW_APPLICATIONS")
    cleaned = raw.filter(...).with_column(...)
    features = cleaned.select(...)

    from snowflake.ml.registry import Registry
    reg = Registry(session=session, database_name="CREDIT_RISK", schema_name="ML")
    mv = reg.get_model("CREDIT_DEFAULT_MODEL").version("v1")
    scored = mv.run(features, function_name="predict_proba")

    scored.write.save_as_table("SCORED_APPLICATIONS", mode="append")
    session.sql("CALL SYSTEM$SEND_EMAIL(...)").collect()
    return "Scoring complete"

session.sproc.register(
    func=monthly_scoring,
    name="MONTHLY_SCORING",
    packages=["snowflake-snowpark-python", "snowflake-ml-python"],
    is_permanent=True,
    stage_location="@ML_STAGE",
    replace=True,
)
```

```sql
CREATE OR REPLACE TASK run_monthly_scoring
  WAREHOUSE = COMPUTE_WH
  SCHEDULE = 'USING CRON 0 6 2 * * America/New_York'   -- 6 AM on the 2nd
AS
  CALL MONTHLY_SCORING();
```

### Task graphs for genuinely separate steps

When you want each step monitored, retried, or branched on independently:

```python
from snowflake.core.task.dagv1 import DAG, DAGTask, DAGOperation
from snowflake.core.task import StoredProcedureCall
from datetime import timedelta

# Register each step as a permanent stored procedure first
clean_sp  = session.sproc.register(func=clean_data, name="CLEAN_DATA",
                                   is_permanent=True, stage_location="@ML_STAGE", replace=True)
score_sp  = session.sproc.register(func=score_data, name="SCORE_DATA",
                                   is_permanent=True, stage_location="@ML_STAGE", replace=True)
notify_sp = session.sproc.register(func=send_notifications, name="NOTIFY",
                                   is_permanent=True, stage_location="@ML_STAGE", replace=True)

dag = DAG(name="monthly_credit_scoring", schedule=timedelta(days=30))
with dag:
    t_clean  = DAGTask("clean",  StoredProcedureCall(clean_sp,  stage_location="@ML_STAGE",
                       packages=["snowflake-snowpark-python"]), warehouse="COMPUTE_WH")
    t_score  = DAGTask("score",  StoredProcedureCall(score_sp,  stage_location="@ML_STAGE",
                       packages=["snowflake-snowpark-python", "snowflake-ml-python"]),
                       warehouse="COMPUTE_WH")
    t_notify = DAGTask("notify", StoredProcedureCall(notify_sp, stage_location="@ML_STAGE",
                       packages=["snowflake-snowpark-python"]), warehouse="COMPUTE_WH")
    t_clean >> t_score >> t_notify

DAGOperation(schema).deploy(dag, mode="orreplace")
```

**Gotcha:** when passing `args` to `StoredProcedureCall`, the `func` parameter must be a **registered stored procedure**, not a plain Python function. Register with `session.sproc.register(...)` first.

---

## 8. Handling Upstream Data Dependencies

**The problem:** a monthly table usually updates on the 2nd, but sometimes it's late. A plain scheduled task would run anyway and either fail or — worse — succeed against stale data.

**The fix: triggered tasks.** The task runs when the data actually arrives, not on a clock.

```sql
-- 1. Create a stream on the upstream table
CREATE OR REPLACE STREAM raw_applications_stream ON TABLE RAW_APPLICATIONS;

-- 2. Create a task with a WHEN clause and NO schedule
CREATE OR REPLACE TASK score_when_data_arrives
  WAREHOUSE = COMPUTE_WH
  WHEN SYSTEM$STREAM_HAS_DATA('raw_applications_stream')
AS
  CALL MONTHLY_SCORING();

ALTER TASK score_when_data_arrives RESUME;
```

| Behavior | Detail |
|----------|--------|
| No polling, no wasted compute | Triggered tasks consume no compute until the event fires |
| Runs the moment data lands | No waiting for the next scheduled window |
| Never runs on stale data | If the stream has no changes, the task is skipped without using compute |
| No double-runs | Only one instance runs at a time; new data arriving mid-run queues the next |
| Health check | If a task hasn't run in 12 hours, Snowflake schedules a check to prevent stream staleness |
| Trigger frequency | At most every 30 seconds by default; can be lowered to 10 via `USER_TASK_MINIMUM_TRIGGER_INTERVAL_IN_SECONDS` |

### Multiple dependencies

```sql
CREATE OR REPLACE TASK score_when_all_ready
  WAREHOUSE = COMPUTE_WH
  WHEN SYSTEM$STREAM_HAS_DATA('applications_stream')
   AND SYSTEM$STREAM_HAS_DATA('bureau_data_stream')
AS
  CALL MONTHLY_SCORING();
```

Swap `AND` for `OR` if either source arriving should trigger the run.

### Serverless variant

```sql
CREATE OR REPLACE TASK score_when_data_arrives
  TARGET_COMPLETION_INTERVAL = '15 MINUTES'
  WHEN SYSTEM$STREAM_HAS_DATA('raw_applications_stream')
AS
  CALL MONTHLY_SCORING();
```

### Converting an existing scheduled task

```sql
ALTER TASK my_task SUSPEND;
ALTER TASK my_task UNSET SCHEDULE;
ALTER TASK my_task MODIFY WHEN SYSTEM$STREAM_HAS_DATA('raw_applications_stream');
ALTER TASK my_task RESUME;
```

In `TASK_HISTORY`, the `SCHEDULED_FROM` column shows `TRIGGER` for event-driven runs.

Streams are supported on tables, views, dynamic tables, Iceberg tables, data shares, and directory tables — not on hybrid tables or external tables.

**Docs:** [Triggered tasks](https://docs.snowflake.com/en/user-guide/tasks-triggered)

---

## 9. Model Monitoring — Built-In Metrics

Snowflake ML Observability provides native, off-the-shelf metrics. No Python implementation required, and no third-party monitoring vendor.

### Creating a monitor

```sql
CREATE MODEL MONITOR credit_default_monitor
  WITH
    MODEL = CREDIT_DEFAULT_MODEL
    VERSION = 'v1'
    FUNCTION = 'predict_proba'
    SOURCE = SCORED_APPLICATIONS
    BASELINE = TRAINING_DATA          -- required for drift metrics
    TIMESTAMP_COLUMN = SCORED_AT
    PREDICTION_SCORE_COLUMNS = (PROB_DEFAULT)
    ACTUAL_CLASS_COLUMNS = (ACTUAL_DEFAULT)
    ID_COLUMNS = (APPLICANT_ID)
    WAREHOUSE = COMPUTE_WH
    REFRESH_INTERVAL = '1 day'
    AGGREGATION_WINDOW = '1 day';
```

### Drift metrics

`MODEL_MONITOR_DRIFT_METRIC(monitor, metric, column, granularity, start, end)`

| Metric | Notes |
|--------|-------|
| `POPULATION_STABILITY_INDEX` | PSI. Computed on a **feature** column, this is what's commonly called CSI |
| `JENSEN_SHANNON` | Distribution divergence |
| `WASSERSTEIN` | Earth mover's distance |
| `DIFFERENCE_OF_MEANS` | Simple mean shift |

Available on any feature column, the prediction column, or the actual column.

### Performance metrics

`MODEL_MONITOR_PERFORMANCE_METRIC(monitor, metric, granularity, start, end)`

| Model type | Metrics |
|-----------|---------|
| **Binary classification** | `ROC_AUC`, `CLASSIFICATION_ACCURACY`, `PRECISION`, `RECALL`, `F1_SCORE` |
| Multi-class | `CLASSIFICATION_ACCURACY`, `MACRO_AVERAGE_PRECISION`, `MACRO_AVERAGE_RECALL`, `MICRO_AVERAGE_PRECISION`, `MICRO_AVERAGE_RECALL` |
| Regression | `RMSE`, `MAE`, `MAPE`, `MSE` |

`ROC_AUC` needs the prediction score and actual class columns; the class-based metrics need prediction class and actual class.

### Statistical metrics

`MODEL_MONITOR_STAT_METRIC(...)` — `COUNT`, `COUNT_NULL`, `MIN`, `MAX`, `AVG`, `SUM` on any column.

### Querying metrics

```sql
-- ROC AUC, daily, over the last 30 days
SELECT * FROM TABLE(MODEL_MONITOR_PERFORMANCE_METRIC(
  'CREDIT_DEFAULT_MONITOR', 'ROC_AUC', '1 DAY',
  DATEADD('DAY', -30, CURRENT_DATE()), CURRENT_DATE()
));

-- PSI on a feature column (CSI equivalent)
SELECT * FROM TABLE(MODEL_MONITOR_DRIFT_METRIC(
  'CREDIT_DEFAULT_MONITOR', 'POPULATION_STABILITY_INDEX', 'FICO_SCORE', '1 DAY',
  DATEADD('DAY', -30, CURRENT_DATE()), CURRENT_DATE()
));
```

Since metrics come back as SQL table functions, you can build custom dashboards in Streamlit or any BI tool, and set alerts on thresholds.

### Segmentation

Monitor performance across subsets of the portfolio — by product, channel, or risk tier:

```sql
CREATE MODEL MONITOR ... SEGMENT_COLUMNS = (PRODUCT_TYPE, ACQUISITION_CHANNEL);
```

```sql
SELECT * FROM TABLE(MODEL_MONITOR_PERFORMANCE_METRIC(
  'CREDIT_DEFAULT_MONITOR', 'ROC_AUC', '1 DAY',
  '2026-01-01'::TIMESTAMP_NTZ, '2026-01-31'::TIMESTAMP_NTZ,
  '{"SEGMENTS": [{"column": "PRODUCT_TYPE", "value": "SECURED"}]}'
));
```

Segment columns must be string categorical (bucket numeric columns first). One segment pair per query.

### Dashboards

**AI & ML » Models** in Snowsight, select a model, then a monitor. Graphs of drift, performance, and volume over time, adapted to the model type. Monitors can be compared across versions.

### Constraints worth knowing before you build

| Constraint | Detail |
|-----------|--------|
| Monitor location | Must live in the same database and schema as the model version |
| One-to-one | Each model version has exactly one monitor; monitors can't be shared |
| **Baseline is set at creation** | Drift metrics require baseline data. **Adding a baseline later requires dropping and recreating the monitor** — decide up front |
| Column types | Timestamp columns must be `TIMESTAMP_NTZ`; prediction and actual columns must be `NUMBER` |
| Data quality | Source data can't contain nulls, NaNs, Inf, or probability scores outside 0–1 — these cause refresh failure and suspension |
| Aggregation window | Minimum 1 day, specified in days |
| Scale limits | 500 features max; 250 monitors per account; 5 segment columns max (each ideally under 25 distinct values) |
| Auto-suspend | Suspends after 5 consecutive refresh failures. Check `DESCRIBE MODEL MONITOR` for `aggregation_status` and `aggregation_last_error`, then `ALTER MODEL MONITOR ... RESUME` |
| Model types | Single-output regression, binary classification, multi-class classification |

### The 18-month label lag

For credit default, ground truth arrives long after the prediction. The monitor handles this in two layers: **prediction and feature drift** give you immediate early warning ("are we approving more than usual?", "has the applicant mix shifted?"), and **performance metrics** are recalculated as actual outcomes land.

**Docs:** [ML Observability](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/model-observability) | [Model monitor functions](https://docs.snowflake.com/en/sql-reference/functions-model-monitors)

---

## 10. Streamlit — End-User Applications

Build web applications hosted in Snowflake for non-technical stakeholders. No separate infrastructure to provision.

- Credit analysts get an approve/decline interface
- Risk managers get portfolio dashboards
- Executives get KPI views
- All powered by the same registered model via SQL

Deploy from a Workspace and the app appears as a standalone Streamlit object — end users see only the interface, not the code.

---

## 11. Connecting Snowflake to GitHub

Snowflake integrates directly with Git repositories. Once connected you can browse branches, fetch the latest code, and **commit and push changes back from Workspaces, Streamlit apps, and Notebooks** — without leaving Snowflake.

**Supported platforms:** GitHub, GitLab, Bitbucket, Azure DevOps, AWS CodeCommit

### Option A: GitHub App (simplest for github.com)

A pre-configured OAuth2 application — no OAuth app registration or redirect URI needed.

```sql
CREATE OR REPLACE API INTEGRATION my_git_api_integration
  API_PROVIDER = git_https_api
  API_ALLOWED_PREFIXES = ('https://github.com')
  API_USER_AUTHENTICATION = (TYPE = SNOWFLAKE_GITHUB_APP)
  ENABLED = TRUE;
```

Then create a Git workspace in Snowsight and authorize through the browser.

> Works with github.com, including GitHub Enterprise Cloud organizations hosted there. GitHub Enterprise Cloud with data residency (`*.ghe.com`) and GitHub Enterprise Server require the OAuth2 path.

### Option B: Personal Access Token

```sql
CREATE OR REPLACE SECRET my_git_secret
  TYPE = password
  USERNAME = '<github_username>'
  PASSWORD = '<personal_access_token>';

CREATE OR REPLACE API INTEGRATION my_git_api_integration
  API_PROVIDER = git_https_api
  API_ALLOWED_PREFIXES = ('https://github.com/<your-org>')
  ALLOWED_AUTHENTICATION_SECRETS = (my_git_secret)
  ENABLED = TRUE;

CREATE OR REPLACE GIT REPOSITORY my_ml_repo
  API_INTEGRATION = my_git_api_integration
  GIT_CREDENTIALS = my_git_secret
  ORIGIN = 'https://github.com/<your-org>/<your-repo>.git';
```

### What you can do once connected

- Fetch branches, tags, and commits
- Browse folders and search files in Snowsight
- Commit and push from Workspaces, Notebooks, and Streamlit apps
- Reference repository files as stored procedure or UDF handlers
- Run `EXECUTE IMMEDIATE` from `.sql` files
- Import repo modules into code running in Snowflake

For repositories behind a firewall, Snowflake supports outbound private link connectivity (Business Critical edition required).

**Docs:** [Using a Git repository in Snowflake](https://docs.snowflake.com/en/developer-guide/git/git-overview)

---

## 12. Version Control Strategy

Two complementary layers:

**Code — Git.** Git is the right tool for code version control, and Snowflake integrates with it directly (section 11). Standard branch, PR, and review workflows apply.

**Models — Model Registry.** Separately, the Registry versions the model artifacts themselves, which Git handles poorly:

| Capability | What it gives you |
|-----------|-------------------|
| Named versions | `v1`, `v2`, `v3` — all callable simultaneously |
| Metrics per version | Compare AUC, KS, Gini across versions at a glance |
| Dependency isolation | Each version records its own package versions |
| Lineage | Which table and columns trained each version |
| Aliases | Point a `PRODUCTION` alias at a version; roll back by re-pointing it |
| Access control | Standard Snowflake RBAC per model |

Use both. A model version's lineage tells you what data produced it; your Git history tells you what code produced it.

---

## 13. Loading Data

For ad-hoc loads of large files, the standard path is stage-then-copy:

```sql
CREATE STAGE IF NOT EXISTS CREDIT_RISK.ML.ML_STAGE
  FILE_FORMAT = (TYPE = CSV FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1);
```

```bash
snow stage copy ./large_bureau_extract.csv @CREDIT_RISK.ML.ML_STAGE
```

```sql
COPY INTO CREDIT_RISK.ML.BUREAU_EXTRACT
  FROM @CREDIT_RISK.ML.ML_STAGE/large_bureau_extract.csv
  ON_ERROR = 'CONTINUE';

-- Inspect rejected rows
SELECT * FROM TABLE(VALIDATE(CREDIT_RISK.ML.BUREAU_EXTRACT, JOB_ID => '_last'));
```

Snowsight also has a browser-based **Load Data** wizard for smaller files — convenient for quick one-offs, but it has a file size ceiling, so use the stage path for large extracts.

Useful options for messy bureau extracts:

| Option | Purpose |
|--------|---------|
| `ON_ERROR = 'CONTINUE'` | Load good rows, skip bad ones |
| `MATCH_BY_COLUMN_NAME = 'CASE_INSENSITIVE'` | Map by header name instead of position |
| `PURGE = TRUE` | Delete staged files after a successful load |
| `FORCE = TRUE` | Reload files even if previously loaded |

If these loads become recurring rather than ad-hoc, Snowpipe can auto-ingest files as they land in cloud storage.

> Current file size limits and the Snowsight wizard's ceiling are worth confirming against the docs at implementation time, since they change.

---

## Snowflake Marketplace — Data Enrichment

Snowflake Marketplace hosts thousands of data products from third-party providers. Depending on your use case, you may find relevant datasets that can enrich your models — for example:

- Credit bureau attributes and scores
- Commercial and consumer credit data
- Alternative data (employment, bank transactions, macroeconomic indicators)

Worth exploring to see what's available for your specific needs. When a relevant dataset exists, it appears as a live table in your account — no file transfers, no integration projects.

---

## Putting It Together

A typical build order:

1. **Define features in the Feature Store** — entities, feature views, managed refresh
2. **Generate a point-in-time-correct training set** — `generate_training_set` with `spine_timestamp_col`
3. **Train in a notebook** — same Python, elastic compute, experiment tracking
4. **Register the model** — one `log_model()` call
5. **Call via SQL** — `MODEL!PREDICT_PROBA(...)` from anywhere
6. **Automate scoring** — `CREATE TASK`, scheduled or triggered on data arrival
7. **Monitor** — `CREATE MODEL MONITOR`, with the baseline set at creation

The three decisions worth making before you build, because they're expensive to change later:

| Decision | Why it matters |
|---|---|
| Set a **monitor baseline** at creation | Adding one later requires dropping and recreating the monitor |
| Use `spine_timestamp_col` on training sets | Without it you leak future feature values into training and the model overstates its performance |
| Pass a **Snowpark** DataFrame as `sample_input_data` | pandas works, but only Snowpark captures data lineage |

---

## External References

- [Snowflake ML overview](https://docs.snowflake.com/en/developer-guide/snowflake-ml/overview)
- [Feature Store](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview) · [Advanced feature engineering](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/advanced-feature-engineering) · [Training and inference](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/modeling)
- [Model Registry](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/overview) · [Pre/post-processing with models](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/custom-processing-with-models)
- [ML Observability](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/model-observability) · [Model monitor functions](https://docs.snowflake.com/en/sql-reference/functions-model-monitors)
- [Snowflake Notebooks](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks)
- [Git integration](https://docs.snowflake.com/en/developer-guide/git/git-overview)
- [Triggered tasks](https://docs.snowflake.com/en/user-guide/tasks-triggered) · [Tasks overview](https://docs.snowflake.com/en/user-guide/tasks-intro)
- [Snowflake Marketplace](https://app.snowflake.com/marketplace)
