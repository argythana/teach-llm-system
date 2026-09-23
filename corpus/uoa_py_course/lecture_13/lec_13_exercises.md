<!-- source: lectures_07_13_pandas_plots_scikit/lecture_13_pipelines_gridsearch_mlflow/practice_exercises/lec_13_exercises.ipynb @ 0cc874704aaa -->

# Lecture 13 Exercises: Pipelines, Hyperparameter Search & MLflow

Practice the automation workflow from Lecture 13 on the **heart-disease** dataset.  Each exercise has a short task statement followed by a code cell for your answer.

**How to use this notebook**

- Required exercises (1.1 - 3.2) map to the lecture's required goals **G1 - G6**.
- Stretch exercises (4.1 - 4.3) map to the optional career-track goals **O1 - O3** and use MLflow.
- Replace every `# Your code here` with your solution. The blank cells run as no-ops so the notebook stays clean.
- Keep `RANDOM_STATE = 42` everywhere you seed a split, search, or estimator.

Run the setup cells first, then work through the exercises in order.

## Setup

Run these three cells before starting. They load the data, make a stratified train/test split, and define the shared preprocessing `ColumnTransformer`.

```python
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

RANDOM_STATE = 42

# Heart-disease dataset (lives in the lecture's reading_material/ folder).
df = (
    pd.read_csv("../reading_material/predict_heart_disease_train.csv")
    .drop(columns=["id"])
    .sample(n=1500, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

# Binary target: 1 = heart disease present, 0 = absent
y = (df["Heart Disease"] == "Presence").astype(int)
X = df.drop(columns=["Heart Disease"])

# Continuous numeric features -> scale; integer-coded categorical features -> one-hot
NUMERIC = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
CATEGORICAL = ["Sex", "Chest pain type", "FBS over 120", "EKG results",
               "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium"]

X.shape, y.mean().round(3)
```

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
X_train.shape, X_test.shape
```

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])
```

## Section 1 - Hyperparameter search

`GridSearchCV` and `RandomizedSearchCV` automate the "try many settings, keep the best" loop using cross-validation instead of a single hand-held validation split.

### Exercise 1.1 [G1]

Tune a `LogisticRegression`'s regularisation strength `C` with **`GridSearchCV`** and **5-fold** CV.

- Grid: `C` in `[0.01, 0.1, 1, 10, 100]`.
- Scale the features first (a bare `LogisticRegression` on unscaled data converges poorly) - fit a `StandardScaler` on `X_train` only, then transform both splits.
- Fit the search on the scaled training data.
- Print `best_params_` and `best_score_` (the mean CV accuracy of the winning `C`).

```python
# Your code here
```

### Exercise 1.2 [G2]

A grid has three hyperparameters:

- `n_estimators`: `[100, 200, 300]`
- `max_depth`: `[None, 5, 10, 20]`
- `min_samples_leaf`: `[1, 2, 5]`

**Part A (by hand, no fitting):** compute the number of candidate combinations and the total number of model fits a 5-fold `GridSearchCV` would do (candidates x folds). Print both numbers.

**Part B:** instead of the full grid, run a **`RandomizedSearchCV`** with `n_iter=10` and `cv=5` over the same grid, using a `RandomForestClassifier`. Print `best_params_`, `best_score_`, and confirm it ran only `10 * 5 = 50` fits.

**Part A** - compute candidates and total fits by hand:

```python
# Your code here
```

**Part B** - run the `RandomizedSearchCV`:

```python
# Your code here
```

## Section 2 - Pipelines

A `Pipeline` chains preprocessing and an estimator into one object. Inside cross-validation the transformers refit on each training fold only, so the test fold never leaks into scaling or encoding.

### Exercise 2.1 [G3]

Build a `Pipeline` that wraps the shared `preprocess` `ColumnTransformer` (StandardScaler on `NUMERIC`, OneHotEncoder on `CATEGORICAL`) followed by a `LogisticRegression`.

- Fit the pipeline on `X_train`, `y_train`.
- Score it on `X_test`, `y_test` and print the test accuracy.
- In the markdown cell that follows, write **one sentence** explaining why the scaler must live inside the pipeline rather than being fit on the full dataset beforehand.

```python
# Your code here
```

**Your one-sentence answer (2.1):** _Write here why the scaler must live inside the pipeline._

### Exercise 2.2 [G4]

Use `GridSearchCV` over a pipeline to compare **two classifier families in one search**, using the `step__param` double-underscore syntax.

- Pipeline: `preprocess` then a `model` step.
- Provide a list of two parameter dicts to the grid:
  - one swapping in `LogisticRegression` (tune `model__C`),
  - one swapping in `RandomForestClassifier` (tune `model__n_estimators`).
- Run with `cv=5`, then print `best_params_` and report **which classifier family won**.

`RandomForestClassifier` is just used here as a strong default tree-ensemble baseline - no theory needed.

```python
# Your code here
```

## Section 3 - MLflow tracking

MLflow records each run's parameters, metrics, and the fitted model so you can compare experiments and reload the best one later. These cells log to a local SQLite backend.

Run this MLflow setup cell once. It points the tracking store at a local SQLite file and creates (or reuses) the exercises experiment.

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow_ex.db")
mlflow.set_experiment("lecture13_exercises")
```

### Exercise 3.1 [G5]

Log a `GridSearchCV` run to MLflow.

- Build a pipeline (`preprocess` + `LogisticRegression`) and tune `model__C` over `[0.1, 1, 10]` with `cv=5`.
- Inside `with mlflow.start_run(run_name="exercise_3_1"):`
  - log the best params with `mlflow.log_params(...)`,
  - compute the **test accuracy** of the best estimator and log it with `mlflow.log_metric("test_accuracy", ...)`,
  - log the fitted best estimator with `mlflow.sklearn.log_model(...)`, passing a **signature** built from `mlflow.models.infer_signature` and `input_example=X_train.iloc[:3]`.
- Keep the returned `model_uri` (you reload it in 3.2).

```python
# Your code here
logged_model_uri = None  # set this to the model_uri you log
```

### Exercise 3.2 [G6]

Reload the model you logged in 3.1 and use it.

- Reload it with `mlflow.sklearn.load_model(logged_model_uri)`.
- Predict on the **first 3 rows** of `X_test` and print the predictions.
- In the markdown cell that follows, name **MLflow's three storage layers**.

```python
# Your code here
```

**Your answer (3.2):** _Name MLflow's three storage layers here._

### Exercise 3.3 [G7]

Use `mlflow.evaluate` (the **static-dataset form**) on the held-out test set, then register the model and set a `champion` alias.

- Fit a `preprocess` + `LogisticRegression` pipeline on the training split.
- Build an eval frame: copy `X_test`, add a `label` column (`y_test`) and a `prediction` column (the pipeline's test predictions), then call `mlflow.evaluate(data=eval_df, predictions="prediction", targets="label", model_type="classifier")`.  Print the accuracy from `result.metrics`.
- Inside a run, log the model with `registered_model_name="heart_disease_ex"`, then use `MlflowClient().set_registered_model_alias(...)` to tag that version as `champion`.

Note: use the **static form** (data + predictions + targets) - the model-URI form imports xgboost in this version and is fragile.

```python
# Your code here
```

## Section 4 - Stretch (career track)

These map to the optional goals **O1 - O3**. They go deeper into MLflow: nested runs, model evaluation and registry aliases, and model serving.

### Exercise 4.1 [O1]

Log a small `C`-sweep as MLflow **parent/child runs** and attach a validation-curve figure.

- Sweep `C` over `[0.01, 0.1, 1, 10, 100]`. For each `C`, get the **mean 5-fold CV accuracy** of a `preprocess` + `LogisticRegression(C=...)` pipeline (use `cross_val_score`).
- Open a parent run `with mlflow.start_run(run_name="exercise_4_1_sweep"):` and for each `C` open a **child** run (`nested=True`) logging that `C` and its CV score.
- Build a matplotlib line plot of CV accuracy vs `C` (log-scaled x-axis) and log it on the parent run with `mlflow.log_figure(fig, "validation_curve.png")`.

```python
# Your code here
```

### Exercise 4.2 [O2] - Demo (not graded)

This is a **fully worked demo**, not a blank exercise. It shows that the registered model's local `predict` returns the same answer the REST `/invocations` endpoint would.

A live `mlflow models serve` process cannot run inside this notebook, so the executable cells load the model with `mlflow.pyfunc.load_model` and feed it the exact `{"dataframe_split": {...}}` payload the REST endpoint expects.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from mlflow import MlflowClient

# Self-contained demo: fit + register a model so this cell runs on its own,
# then tag the version "demo_champion".
demo_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
]).fit(X_train, y_train)

with mlflow.start_run(run_name="exercise_4_3_demo"):
    demo_sig = mlflow.models.infer_signature(X_train, demo_pipe.predict(X_train))
    mlflow.sklearn.log_model(
        demo_pipe, name="model", signature=demo_sig,
        input_example=X_train.iloc[:3], registered_model_name="heart_disease_demo",
    )
demo_version = MlflowClient().get_latest_versions("heart_disease_demo")[0].version
MlflowClient().set_registered_model_alias("heart_disease_demo", "demo_champion", version=demo_version)

# Load the registered model as a generic pyfunc model (same wrapper the server uses).
pyfunc_model = mlflow.pyfunc.load_model("models:/heart_disease_demo@demo_champion")

# Build the REST payload from 2 test rows: {"dataframe_split": {"columns": [...], "data": [...]}}
two_rows = X_test.iloc[:2]
payload = {
    "dataframe_split": {
        "columns": list(two_rows.columns),
        "data": two_rows.values.tolist(),
    }
}

# The /invocations endpoint reconstructs a DataFrame from dataframe_split, then predicts.
# We restore the original per-column dtypes so the model signature validates cleanly
# (the REST server does the same JSON-to-typed-DataFrame step internally).
payload_df = pd.DataFrame(
    payload["dataframe_split"]["data"],
    columns=payload["dataframe_split"]["columns"],
).astype(two_rows.dtypes)
local_predictions = pyfunc_model.predict(payload_df)
print("local pyfunc predictions:", local_predictions)
print("these match what the REST /invocations endpoint would return for the same payload")
```

**Serving the same model over REST (run in a terminal, not here):**

```bash
# 1. Start the model server (separate terminal). --no-conda reuses the current env.
mlflow models serve -m "models:/heart_disease_demo@demo_champion" --port 5002 --no-conda

# 2. POST the same dataframe_split payload to the /invocations endpoint.
curl -X POST http://127.0.0.1:5002/invocations \
     -H "Content-Type: application/json" \
     -d '{"dataframe_split": {"columns": ["Age", "Sex", "..."], "data": [[...], [...]]}}'

# Response: {"predictions": [0, 1]} - the same values pyfunc_model.predict printed above.
```

The model **signature** logged in 4.2 lets the server validate incoming columns and types before predicting, rejecting malformed requests.

## Cleanup

The MLflow stores (`mlflow_ex.db`, `mlruns/`, `mlartifacts/`) are local scratch and gitignored. You can delete them once you are done exploring the runs.
