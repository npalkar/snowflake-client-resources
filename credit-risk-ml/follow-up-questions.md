# Follow-Up: Open Questions

Answers to questions raised during the modeling walkthrough.

---

## 1. Connecting Snowflake to GitHub

Snowflake integrates directly with Git repositories. Once connected, you can browse branches, fetch the latest code, and **commit and push changes back to the remote repository from Workspaces, Streamlit apps, and Notebooks** — all without leaving Snowflake.

**Supported platforms:** GitHub, GitLab, Bitbucket, Azure DevOps, AWS CodeCommit

### Option A: GitHub App (simplest for github.com)

The Snowflake GitHub App is a pre-configured OAuth2 application — no OAuth app registration or redirect URI needed.

```sql
CREATE OR REPLACE API INTEGRATION my_git_api_integration
  API_PROVIDER = git_https_api
  API_ALLOWED_PREFIXES = ('https://github.com')
  API_USER_AUTHENTICATION = (TYPE = SNOWFLAKE_GITHUB_APP)
  ENABLED = TRUE;
```

Then create a Git workspace in Snowsight and authorize through the browser.

> Works with github.com, including GitHub Enterprise Cloud organizations hosted there. GitHub Enterprise Cloud with data residency (`*.ghe.com`) and GitHub Enterprise Server require the OAuth2 path instead.

### Option B: Personal Access Token

```sql
-- Store credentials as a secret
CREATE OR REPLACE SECRET my_git_secret
  TYPE = password
  USERNAME = '<github_username>'
  PASSWORD = '<personal_access_token>';

-- Create the API integration
CREATE OR REPLACE API INTEGRATION my_git_api_integration
  API_PROVIDER = git_https_api
  API_ALLOWED_PREFIXES = ('https://github.com/<your-org>')
  ALLOWED_AUTHENTICATION_SECRETS = (my_git_secret)
  ENABLED = TRUE;

-- Clone the repository into Snowflake
CREATE OR REPLACE GIT REPOSITORY my_ml_repo
  API_INTEGRATION = my_git_api_integration
  GIT_CREDENTIALS = my_git_secret
  ORIGIN = 'https://github.com/<your-org>/<your-repo>.git';
```

### What you can do once connected

- Fetch branches, tags, and commits from the remote repository
- Browse folders and search files in Snowsight
- Commit and push from Workspaces, Notebooks, and Streamlit apps
- Reference repository files as stored procedure or UDF handlers
- Run `EXECUTE IMMEDIATE` from `.sql` files in the repo
- Import repo modules into code running in Snowflake

### Private network option

For repositories behind a firewall, Snowflake supports outbound private link connectivity (requires Business Critical edition). See [Connect to a Git repository over a private network](https://docs.snowflake.com/en/developer-guide/git/git-setting-up-private).

**Docs:** [Using a Git repository in Snowflake](https://docs.snowflake.com/en/developer-guide/git/git-overview)

---

## 2. Model Monitoring — Built-In Metrics

Snowflake ML Observability provides **native, off-the-shelf metrics** — no Python required. You create a monitor, and the metrics are queryable via SQL table functions.

### Drift metrics

`MODEL_MONITOR_DRIFT_METRIC(monitor_name, metric_name, column_name, granularity, start_time, end_time)`

| Metric | Notes |
|--------|-------|
| `POPULATION_STABILITY_INDEX` | PSI — and when computed on a **feature** column, this is what's commonly called CSI (Characteristic Stability Index) |
| `JENSEN_SHANNON` | Distribution divergence |
| `WASSERSTEIN` | Earth mover's distance |
| `DIFFERENCE_OF_MEANS` | Simple mean shift |

Drift can be computed on **any feature column, the prediction column, or the actual column**.

### Performance metrics

`MODEL_MONITOR_PERFORMANCE_METRIC(monitor_name, metric_name, granularity, start_time, end_time)`

| Model type | Available metrics |
|-----------|-------------------|
| **Binary classification** | `ROC_AUC`, `CLASSIFICATION_ACCURACY`, `PRECISION`, `RECALL`, `F1_SCORE` |
| Multi-class | `CLASSIFICATION_ACCURACY`, `MACRO_AVERAGE_PRECISION`, `MACRO_AVERAGE_RECALL`, `MICRO_AVERAGE_PRECISION`, `MICRO_AVERAGE_RECALL` |
| Regression | `RMSE`, `MAE`, `MAPE`, `MSE` |

### Statistical metrics

`MODEL_MONITOR_STAT_METRIC(...)` — `COUNT`, `COUNT_NULL`, `MIN`, `MAX`, `AVG`, `SUM` on any column.

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

### Querying metrics

```sql
-- ROC AUC over the last 30 days, daily granularity
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

Because metrics are returned as SQL table functions, you can build custom dashboards in Streamlit, PowerBI, or any BI tool — or set up alerts and notifications on thresholds.

### Segmentation

Monitor performance across subsets of your portfolio (e.g., by product, channel, or risk tier) using `SEGMENT_COLUMNS`:

```sql
CREATE MODEL MONITOR ... SEGMENT_COLUMNS = (PRODUCT_TYPE, ACQUISITION_CHANNEL);
```

Then query a specific segment:

```sql
SELECT * FROM TABLE(MODEL_MONITOR_PERFORMANCE_METRIC(
  'CREDIT_DEFAULT_MONITOR', 'ROC_AUC', '1 DAY',
  '2026-01-01'::TIMESTAMP_NTZ, '2026-01-31'::TIMESTAMP_NTZ,
  '{"SEGMENTS": [{"column": "PRODUCT_TYPE", "value": "SECURED"}]}'
));
```

### Dashboards in Snowsight

Navigate to **AI & ML » Models**, select a model, then select a monitor to see graphs of drift, performance, and volume over time. Graphs adapt to the model type. You can compare monitors across model versions.

### Constraints worth knowing up front

| Constraint | Detail |
|-----------|--------|
| Monitor location | Must be in the same database and schema as the model version |
| One-to-one | Each model version has exactly one monitor; monitors can't be shared |
| Baseline is set at creation | Drift metrics require baseline data. **To add a baseline later you must drop and recreate the monitor** |
| Column types | Timestamp columns must be `TIMESTAMP_NTZ`; prediction and actual columns must be `NUMBER` |
| Data quality | Source data can't contain nulls, NaNs, Inf, or probability scores outside 0-1 — these cause refresh failure and suspension |
| Aggregation window | Minimum 1 day, specified in days |
| Scale limits | Max 500 features monitored; max 250 monitors per account; max 5 segment columns (each with <25 unique values recommended) |
| Auto-suspend | Monitors suspend after 5 consecutive refresh failures. Check `DESCRIBE MODEL MONITOR` for `aggregation_status` and `aggregation_last_error` |
| Model types | Single-output regression, binary classification, multi-class classification |

**Docs:** [ML Observability](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/model-observability) | [Model monitor functions](https://docs.snowflake.com/en/sql-reference/functions-model-monitors)

---

## 3. Scheduling Multi-Step Python Scripts

Your production flow (data cleaning → transformation → scoring → logging → notifications) doesn't need to be rewritten as SQL. Wrap Python in stored procedures and orchestrate them with tasks.

### Single Python script on a schedule

```python
def monthly_scoring(session):
    # Step 1: clean
    raw = session.table("RAW_APPLICATIONS")
    cleaned = raw.filter(...).with_column(...)

    # Step 2: transform
    features = cleaned.select(...)

    # Step 3: score using the registered model
    from snowflake.ml.registry import Registry
    reg = Registry(session=session, database_name="CREDIT_RISK", schema_name="ML")
    mv = reg.get_model("CREDIT_DEFAULT_MODEL").version("v1")
    scored = mv.run(features, function_name="predict_proba")

    # Step 4: write results
    scored.write.save_as_table("SCORED_APPLICATIONS", mode="append")

    # Step 5: log / notify
    session.sql("CALL SYSTEM$SEND_EMAIL(...)").collect()

    return "Scoring complete"

# Register as a permanent stored procedure
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

ALTER TASK run_monthly_scoring RESUME;
```

### Multi-step pipelines as a task graph (DAG)

For genuinely separate steps you want to monitor, retry, or branch on independently, use the Python DAG API:

```python
from snowflake.core.task.dagv1 import DAG, DAGTask, DAGOperation
from snowflake.core.task import StoredProcedureCall
from datetime import timedelta

# Register each step as a permanent stored procedure first
clean_sp = session.sproc.register(func=clean_data, name="CLEAN_DATA",
                                  is_permanent=True, stage_location="@ML_STAGE", replace=True)
score_sp = session.sproc.register(func=score_data, name="SCORE_DATA",
                                  is_permanent=True, stage_location="@ML_STAGE", replace=True)
notify_sp = session.sproc.register(func=send_notifications, name="NOTIFY",
                                   is_permanent=True, stage_location="@ML_STAGE", replace=True)

dag = DAG(name="monthly_credit_scoring", schedule=timedelta(days=30))
with dag:
    t_clean  = DAGTask("clean",  StoredProcedureCall(clean_sp,  stage_location="@ML_STAGE",
                       packages=["snowflake-snowpark-python"]), warehouse="COMPUTE_WH")
    t_score  = DAGTask("score",  StoredProcedureCall(score_sp,  stage_location="@ML_STAGE",
                       packages=["snowflake-snowpark-python", "snowflake-ml-python"]), warehouse="COMPUTE_WH")
    t_notify = DAGTask("notify", StoredProcedureCall(notify_sp, stage_location="@ML_STAGE",
                       packages=["snowflake-snowpark-python"]), warehouse="COMPUTE_WH")
    t_clean >> t_score >> t_notify

DAGOperation(schema).deploy(dag, mode="orreplace")
```

Each step runs in order, gets its own execution history, and can be retried independently.

> **Gotcha:** when passing `args` to `StoredProcedureCall`, the `func` parameter must be a **registered stored procedure**, not a plain Python function. Register with `session.sproc.register(...)` first.

**Docs:** [Writing stored procedures with Python](https://docs.snowflake.com/en/developer-guide/stored-procedure/python/procedure-python-overview) | [Managing tasks and task graphs with Python](https://docs.snowflake.com/en/developer-guide/snowflake-python-api/snowflake-python-managing-tasks)

---

## 4. Upstream Data Dependencies — Avoiding Silent Failures

**The problem:** your monthly table usually updates on the 2nd, but sometimes it's late. A plain scheduled task would run anyway and either fail or — worse — succeed against stale data.

**The fix: triggered tasks.** Instead of running on a clock, the task runs *when the data actually arrives*.

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

### Why this solves the problem

| Behavior | Detail |
|----------|--------|
| No polling, no wasted compute | Triggered tasks consume **no compute resources until the event fires** |
| Runs the moment data lands | No waiting for the next scheduled window |
| Never runs on stale data | If the stream has no changes, the task is skipped without using compute |
| No double-runs | Snowflake ensures only one instance runs at a time; if new data arrives mid-run, the next run queues |
| Health check | If a task hasn't run in 12 hours, Snowflake schedules a check to prevent stream staleness |
| Trigger frequency | At most every 30 seconds by default; can be lowered to 10 seconds via `USER_TASK_MINIMUM_TRIGGER_INTERVAL_IN_SECONDS` |

### Multiple dependencies

Require **all** upstream tables to have new data before scoring:

```sql
CREATE OR REPLACE TASK score_when_all_ready
  WAREHOUSE = COMPUTE_WH
  WHEN SYSTEM$STREAM_HAS_DATA('applications_stream')
   AND SYSTEM$STREAM_HAS_DATA('bureau_data_stream')
AS
  CALL MONTHLY_SCORING();
```

Swap `AND` for `OR` if either source arriving should trigger the run.

### Serverless option

Skip warehouse management entirely — Snowflake sizes the compute for you:

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

### Monitoring

In `TASK_HISTORY` (both `INFORMATION_SCHEMA` and `ACCOUNT_USAGE`), the `SCHEDULED_FROM` column shows `TRIGGER` for triggered task runs — so you can distinguish event-driven runs from scheduled ones.

**Note:** streams are supported on tables, views, dynamic tables, Iceberg tables, data shares, and directory tables. Not supported on hybrid tables or streams on external tables.

**Docs:** [Triggered tasks](https://docs.snowflake.com/en/user-guide/tasks-triggered)

---

## 5. Preprocessing Pipelines in the Model Registry

Yes — you can register a preprocessing pipeline together with the model so a single call does both.

### Option A: sklearn Pipeline (recommended)

The registry natively supports `sklearn.pipeline.Pipeline`. Log the whole pipeline as one object — preprocessing and scoring happen together on every call.

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

Now raw, unprocessed input goes in and a score comes out — the transformation is baked in:

```sql
SELECT CREDIT_RISK.ML.CREDIT_DEFAULT_PIPELINE!PREDICT_PROBA(<raw columns>) FROM NEW_APPLICATIONS;
```

You can mix scikit-learn preprocessing with an XGBoost model in the same pipeline.

### Option B: CustomModel with multiple objects

If your preprocessing isn't expressible as an sklearn Pipeline, or you need to chain multiple models, use `CustomModel` with a `ModelContext`:

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

my_model = CreditScoringModel(mc)

mv = reg.log_model(
    my_model,
    model_name="CREDIT_DEFAULT_CUSTOM",
    version_name="v1",
    sample_input_data=X_test.head(10),
    conda_dependencies=["scikit-learn", "xgboost"],
)
```

Supported model objects passed into the context are serialized automatically — you don't need to pickle them yourself.

> Always access model objects through `self.context[...]` inside the class rather than assigning them to `self` in `__init__`. Direct assignment captures a second copy in a closure and significantly inflates the serialized model size.

> Inference methods must handle **multi-row DataFrames** — server-side batching can combine single-record requests from multiple sources into one DataFrame.

### Ad-hoc scenario analysis

For the "what if we changed X, Y, Z" workflow — no need to persist anything. Call the pipeline against an inline or temporary dataset and read the results directly. Nothing is written unless you explicitly save it.

### Bringing helper code along

If your preprocessing depends on custom modules, package them with `code_paths`:

```python
mv = reg.log_model(
    my_model,
    model_name="CREDIT_DEFAULT_CUSTOM",
    version_name="v1",
    code_paths=["src/preprocessing_utils"],   # import with: import preprocessing_utils
    ...
)
```

**Docs:** [Pre-processing and post-processing with models](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/custom-processing-with-models) | [scikit-learn in the registry](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/built-in-models/scikit-learn)

---

## 6. Version Control in Snowflake

Two complementary layers:

### Code versioning — Git integration

Git is the right tool for code version control, and Snowflake integrates with it directly (see section 1). Commit and push from Workspaces, Notebooks, and Streamlit apps; standard branch, PR, and review workflows apply.

### Model versioning — Model Registry

Separately, the Model Registry versions the **model artifacts** themselves — something Git handles poorly:

| Capability | What it gives you |
|-----------|-------------------|
| Named versions | `v1`, `v2`, `v3` — all callable simultaneously |
| Metrics per version | Compare AUC, KS, Gini across versions at a glance |
| Dependency isolation | Each version records its own package versions; no conflicts between models |
| Lineage | Which table and columns trained each version |
| Aliases | Point a `PRODUCTION` alias at a version; roll back by re-pointing the alias |
| Access control | Standard Snowflake RBAC per model |

Use both: **Git for code, Registry for models.** A model version's lineage tells you what data produced it; your Git history tells you what code produced it.

---

## 7. Ad-Hoc Ingestion of Large CSVs

For loading large CSV files outside of a regular pipeline, the standard path is stage-then-copy:

```sql
-- 1. Create a stage (once)
CREATE STAGE IF NOT EXISTS CREDIT_RISK.ML.ML_STAGE
  FILE_FORMAT = (TYPE = CSV FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1);
```

```bash
# 2. Upload from your machine (Snowflake CLI or SnowSQL)
snow stage copy ./large_bureau_extract.csv @CREDIT_RISK.ML.ML_STAGE
```

```sql
-- 3. Load into a table
COPY INTO CREDIT_RISK.ML.BUREAU_EXTRACT
  FROM @CREDIT_RISK.ML.ML_STAGE/large_bureau_extract.csv
  ON_ERROR = 'CONTINUE';           -- or 'ABORT_STATEMENT' for strict loads

-- 4. Inspect any rejected rows
SELECT * FROM TABLE(VALIDATE(CREDIT_RISK.ML.BUREAU_EXTRACT, JOB_ID => '_last'));
```

Snowsight also has a browser-based **Load Data** wizard for smaller files, which is convenient for quick one-offs but has a file size ceiling — use the stage path above for large extracts.

Useful options for messy bureau extracts:

| Option | Purpose |
|--------|---------|
| `ON_ERROR = 'CONTINUE'` | Load good rows, skip bad ones |
| `MATCH_BY_COLUMN_NAME = 'CASE_INSENSITIVE'` | Map columns by header name instead of position |
| `PURGE = TRUE` | Delete staged files after a successful load |
| `FORCE = TRUE` | Reload files even if previously loaded |

If these extracts become recurring rather than ad-hoc, Snowpipe can auto-ingest files as they land in cloud storage — worth revisiting once the pattern is established.

> Current file size limits and the Snowsight wizard's ceiling are worth confirming against the docs at implementation time, since they change.

---

## Summary

| Question | Answer |
|----------|--------|
| Connect to GitHub? | Yes — Git integration with commit/push from Workspaces, Notebooks, and Streamlit. GitHub App is the simplest setup |
| Built-in monitoring metrics (PSI, CSI, AUC, precision/recall)? | Yes — all native and queryable via SQL. PSI on a feature column is CSI |
| Schedule multi-step Python? | Yes — Python stored procedures + tasks, or task graphs for multi-step DAGs |
| Handle late upstream data? | Yes — triggered tasks fire on data arrival, not on a clock. No compute used while waiting |
| Register a preprocessing pipeline with the model? | Yes — sklearn Pipeline natively, or CustomModel for anything more complex |
| Version control? | Git for code, Model Registry for model artifacts |
| Large CSV ad-hoc loads? | Stage + `COPY INTO`; Snowsight wizard for smaller files |
