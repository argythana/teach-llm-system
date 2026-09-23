<!-- source: lectures_07_13_pandas_plots_scikit/lecture_13_pipelines_gridsearch_mlflow/reading_material/lec_13c_mlflow_tracking_basics.ipynb @ 0cc874704aaa -->

# Lecture 13c: Tracking Experiments with MLflow

**Status: Mandatory reading.**

In lec_13a and lec_13b you ran dozens of `GridSearchCV` configurations across several classifier families. By the end, a fair question is hard to answer from memory alone: **which configuration gave which score, and with which preprocessing?** Scrolling back through notebook outputs does not scale, and re-running a notebook quietly overwrites the previous results. MLflow fixes this by **recording every run** — its parameters, its metrics, and the fitted model itself — into a store you can query later, so results stay reproducible and comparable instead of vanishing the moment the cell re-executes.

**What this notebook covers**

- **What MLflow is** and how to stand up a local tracking server (`mlflow server`) vs a read-only viewer (`mlflow ui`).
- **The three storage layers** MLflow keeps: backend store, artifact store, and model registry.
- **Logging a run** — parameters, metrics, and the fitted model for a `GridSearchCV` over a `Pipeline` (goal G5).
- **Comparing many runs headlessly** with `mlflow.search_runs`, exactly what the UI table shows visually.
- **Reloading a logged model** to predict with it, and seeing where each piece lives on disk (goal G6).
- **Autologging** (`mlflow.autolog`), **tags** (`set_tag`, searchable), and **logging a figure** to the artifact store.

**Forward pointer:** this is the first MLflow notebook and is mandatory. The next mandatory notebook `lec_13d` adds `mlflow.evaluate`, validation gates, and the model registry (champion/challenger). The optional career-track notebooks `lec_13e` and `lec_13f` go deeper — parent/child sweep logging and REST serving via `mlflow models serve`.

## What MLflow is, and the server setup

MLflow is an open-source tool for **tracking machine-learning experiments**. Each time you train a model you open a *run*; into that run you log the inputs you chose (parameters), the numbers you got back (metrics), and the trained model itself. MLflow stores all of this so you can come back days later and compare runs side by side.

**The three storage layers** (each holds a different kind of thing):

- **Backend store** — the small structured data: parameters, metrics, tags, run timestamps. Here it is a **SQLite database file**. This is what the comparison table reads from.
- **Artifact store** — the larger files: the serialised model, plots, any logged file. Here it is a **local folder** (`./mlartifacts`).
- **Model registry** — a catalogue of **named, versioned models** with aliases like `champion`. Covered next in the mandatory lec_13d; we only name it here.

Start a full server once, in a **separate terminal** (it keeps running while you work):

```bash
# one-time, in a separate terminal:
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
# then browse runs at http://127.0.0.1:5000
# read-only alternative (viewer only, no REST API / no serving):
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**`mlflow ui` vs `mlflow server`** — both open the same web table of runs, but:

- **`mlflow ui`** is a **read-only viewer** pointed at an existing store. Good for browsing results.
- **`mlflow server`** is the **full server**: the web UI **plus** a REST API and model-serving endpoints. Use this when other processes (or notebooks on another machine) need to log to or serve from the store.

You do **not** need the server running to follow this notebook. The cells below log **directly** to the same SQLite file the server reads, so everything we log shows up in the UI the moment you start the server.

## Imports and connecting to the tracking store

One imports cell, then point MLflow at our SQLite backend and name our experiment. An *experiment* is just a named bucket that groups related runs.

```python
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import requests
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
```

```python
# Point MLflow at our SQLite backend store and name the experiment.
# These cells write to the SAME sqlite file the `mlflow server` command reads,
# so every run logged below appears in the UI once the server is running.
# Prefer a running `mlflow server` so runs show up live in the web UI; if it is not
# running, fall back to the same SQLite file directly so the notebook still runs.
MLFLOW_UI = "http://127.0.0.1:5000"

def mlflow_server_running(uri=MLFLOW_UI, timeout=2):
    """True if an `mlflow server` answers on /health, else False."""
    try:
        return requests.get(f"{uri}/health", timeout=timeout).status_code == 200
    except requests.exceptions.RequestException:
        return False

if mlflow_server_running():
    mlflow.set_tracking_uri(MLFLOW_UI)
    print(f"MLflow server is UP at {MLFLOW_UI} -- runs appear live in the web UI.")
    print(f"Open {MLFLOW_UI} in your browser to watch this experiment fill up.")
else:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    print("MLflow server is NOT running -- logging to the local SQLite store 'mlflow.db'.")
    print("To explore these runs in the UI, open a terminal in this folder and run:")
    print("    mlflow server --backend-store-uri sqlite:///mlflow.db \\")
    print("                  --default-artifact-root ./mlartifacts --port 5000")
    print("then open http://127.0.0.1:5000  (re-run this cell to switch to the server).")
mlflow.set_experiment("lecture13_tracking")
```

## The data and the pipeline

Same heart-disease data and the same `Pipeline(preprocess, clf)` pattern you built in lec_13b — load, split with stratification, and assemble the `ColumnTransformer`. Kept brief here; lec_13b explains every step.

```python
# Heart-disease dataset (committed alongside this notebook). 630k rows -> sample for fast teaching.
df = (
    pd.read_csv("predict_heart_disease_train.csv")
    .drop(columns=["id"])
    .sample(n=1500, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

# Binary target: 1 = heart disease present, 0 = absent
y = (df["Heart Disease"] == "Presence").astype(int)
X = df.drop(columns=["Heart Disease"])

# Continuous numeric features -> scale; integer-coded categorical features -> one-hot encode
NUMERIC = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
CATEGORICAL = ["Sex", "Chest pain type", "FBS over 120", "EKG results",
               "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium"]

X.shape, y.mean().round(3)
```

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# The leakage-free ColumnTransformer from lec_13b: scale numerics, one-hot the categoricals.
preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])

X_train.shape, X_test.shape
```

## Logging ONE run

Goal G5. We open a run with `with mlflow.start_run(...):`, run a small `GridSearchCV` over a logistic-regression pipeline inside it, then log:

- **`log_params`** — the winning hyperparameters (`gs.best_params_`).
- **`log_metric`** — the cross-validated accuracy (`gs.best_score_`) and the held-out test accuracy.
- **`log_model`** — the fitted best estimator, with an inferred *signature* (the input/output schema) and a small `input_example` so the UI can show the expected columns.

Always use the **`with` context manager** — it opens the run and, crucially, **closes it** when the block ends even if an error is raised.

```python
# A logistic-regression pipeline + a tiny grid over the inverse-regularisation strength C.
logreg_pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])

logreg_grid = {"clf__C": [0.1, 1.0, 10.0]}
```

```python
with mlflow.start_run(run_name="logreg_baseline"):
    
    gs = GridSearchCV(
        logreg_pipe, 
        logreg_grid, 
        cv=5, 
        scoring="accuracy")
    
    gs.fit(X_train, y_train)

    test_acc = accuracy_score(
        y_test, 
        gs.best_estimator_.predict(X_test))

    # 1) parameters: the winning hyperparameters
    mlflow.log_params(gs.best_params_)

    # 2) metrics: cross-validated score and held-out test score
    mlflow.log_metric("cv_accuracy", gs.best_score_)
    mlflow.log_metric("test_accuracy", test_acc)

    # 3) the fitted model, with a schema (signature) and an example input row
    sig = mlflow.models.infer_signature(X_train, gs.best_estimator_.predict(X_train))
    
    mlflow.sklearn.log_model(
        gs.best_estimator_, name="model", signature=sig, input_example=X_train.iloc[:3]
    )

print(f"logreg_baseline  cv={gs.best_score_:.3f}  test={test_acc:.3f}")
```

## Logging a FEW more runs in a loop

One run is not comparable to anything. To make the comparison meaningful we log **a few more runs**, each a different classifier family with its own small grid, and each in its **own `start_run`**. This is the everyday MLflow workflow: try several ideas, let the store remember them all.

`RandomForestClassifier` joins as a candidate here — treat it simply as a strong default tree-ensemble baseline; the theory is not the point in this notebook, it is just another thing to search over.

```python
# Each entry: a run name -> (pipeline, parameter grid). Three different classifier families.
candidates = {
    "svc_rbf": (
        Pipeline([("preprocess", preprocess),
                  ("clf", SVC(random_state=RANDOM_STATE))]),
        {"clf__C": [0.5, 1.0, 5.0], "clf__gamma": ["scale", "auto"]},
    ),
    "random_forest": (
        Pipeline([("preprocess", preprocess),
                  ("clf", RandomForestClassifier(random_state=RANDOM_STATE))]),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [4, 8]},
    ),
    "logreg_l1": (
        Pipeline([("preprocess", preprocess),
                  ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                                             max_iter=1000, random_state=RANDOM_STATE))]),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
}
```

```python
for run_name, (pipe, grid) in candidates.items():
    with mlflow.start_run(run_name=run_name):
        gs = GridSearchCV(pipe, grid, cv=5, scoring="accuracy")
        gs.fit(X_train, y_train)

        test_acc = accuracy_score(y_test, gs.best_estimator_.predict(X_test))

        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("cv_accuracy", gs.best_score_)
        mlflow.log_metric("test_accuracy", test_acc)
        sig = mlflow.models.infer_signature(X_train, gs.best_estimator_.predict(X_train))
        mlflow.sklearn.log_model(
            gs.best_estimator_, name="model", signature=sig, input_example=X_train.iloc[:3]
        )

        print(f"{run_name:14s}  cv={gs.best_score_:.3f}  test={test_acc:.3f}")
```

## Comparing runs headlessly

`mlflow.search_runs` returns **a pandas DataFrame** with one row per run — its parameters, its metrics, its timestamps. This is exactly the table the MLflow UI shows you visually; here we get it as a DataFrame and sort it ourselves. Sorting by `metrics.test_accuracy` answers the question this whole notebook started with: *which run won?*

```python
runs = mlflow.search_runs(experiment_names=["lecture13_tracking"])

# Keep the readable columns and sort best-first by held-out test accuracy.
comparison = (
    runs[["tags.mlflow.runName", "metrics.cv_accuracy", "metrics.test_accuracy"]]
    .sort_values("metrics.test_accuracy", ascending=False)
    .reset_index(drop=True)
)
comparison
```

## See these runs in the MLflow UI

The DataFrame above is the headless version of the UI's run table. To explore the **same runs** visually:

1. If the server is not already running, start it in a terminal **in this folder**:  
   `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 5000`
2. Open **http://127.0.0.1:5000** and click the **`lecture13_tracking`** experiment.
3. Tick two or more runs and press **Compare** to see their parameters and metrics side by side.

Re-run the setup cell after starting the server and every new run appears here live.

## Reloading a logged model and predicting

Goal G6. A logged model is not just a number in a table — the **fitted estimator itself** sits in the artifact store and can be reloaded. We take the best run from the comparison above, build its model URI as `runs:/<run_id>/model`, load it with `mlflow.sklearn.load_model`, and predict on a few test rows. The reloaded model includes the full preprocessing pipeline, so it accepts the **raw** `X_test` columns — no manual scaling or encoding needed.

```python
# Grab the run_id of the best run, then reload its model from the artifact store.
best_run_id = runs.sort_values("metrics.test_accuracy", ascending=False).iloc[0]["run_id"]
model_uri = f"runs:/{best_run_id}/model"

reloaded_model = mlflow.sklearn.load_model(model_uri)
model_uri
```

```python
# Predict on the first 5 raw test rows; confirm the reloaded model matches the live one's choices.
reloaded_preds = reloaded_model.predict(X_test.iloc[:5])
```

```python

pd.DataFrame({
    "reloaded_prediction": reloaded_preds,
    "true_label": y_test.iloc[:5].values,
})
```

## Autologging: let MLflow log for you

Every run above logged params, metrics, and the model with **explicit `log_*` calls**. MLflow can do most of that automatically. Call `mlflow.sklearn.autolog()` once, and from then on **every `fit`** captures the estimator's parameters, training metrics, and the fitted model on its own — no manual logging inside the run.

**Autolog vs manual logging** — pick by what you need:

- **Autolog** — fastest path. One line, and you stop forgetting to log things. Best when you just want a faithful record of standard sklearn fits.
- **Manual `log_*`** — full control. You choose exactly which params, custom metrics (like our held-out `test_accuracy`), tags, and artifacts to record. Best when the thing you care about is not something sklearn computes during `fit`.

The two combine: turn on autolog for the baseline capture, then add manual `log_metric` / `log_figure` calls for anything extra.

```python
# Turn on sklearn autologging. From now on, every fit auto-captures params + metrics + model.
mlflow.sklearn.autolog()

# Fit a model inside a run WITHOUT a single manual log_* call.
with mlflow.start_run(run_name="autolog_logreg") as auto_run:
    auto_pipe = Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    auto_pipe.fit(X_train, y_train)

auto_run_id = auto_run.info.run_id
auto_run_id
```

```python
# Read the autolog run back: params and metrics MLflow captured with NO manual calls.
from mlflow import MlflowClient

client = MlflowClient()
auto_run_data = client.get_run(auto_run_id).data

print("Auto-captured params (sample):")
for k in ["clf__C", "clf__max_iter", "clf__penalty"]:
    print(f"  {k} = {auto_run_data.params.get(k)}")

print("\nAuto-captured metrics (sample):")
for k in ["training_accuracy_score", "training_f1_score"]:
    print(f"  {k} = {auto_run_data.metrics.get(k)}")

# Turn autolog back off so the remaining cells log explicitly, as before.
mlflow.sklearn.autolog(disable=True)
```

## Tagging a run

A **tag** is a short, named note attached to a run — the model family, the dataset version, who ran it, "candidate for production". Unlike a metric, a tag is free-form text and is meant for **finding** runs later, not for ranking them.

- Set one with `mlflow.set_tag(key, value)` on the **active** run, or `client.set_tag(run_id, key, value)` on **any** run by id.
- Tags are **searchable**: `search_runs` accepts a `filter_string` like `tags.model_family = 'logistic_regression'`, exactly like the UI's search box.

Here we tag the best run from the comparison so we can find it again by note rather than by remembering its run id.

```python
# Tag the best run with a short note (model family + a "selected" marker), then find it by tag.
client.set_tag(best_run_id, "model_family", "support_vector_machine")
client.set_tag(best_run_id, "note", "best_test_accuracy")

# search_runs can now filter on that tag, just like the UI search box.
tagged = mlflow.search_runs(
    experiment_names=["lecture13_tracking"],
    filter_string="tags.note = 'best_test_accuracy'",
)
tagged[["run_id", "tags.mlflow.runName", "tags.model_family", "metrics.test_accuracy"]]
```

## Logging a figure to the artifact store

Params and metrics are numbers in the backend store. A **plot is a file** — it belongs in the **artifact store** instead. `mlflow.log_figure(fig, "confusion_matrix.png")` saves a matplotlib figure straight into the active run's artifacts, no temp file needed.

We reopen the best run with `start_run(run_id=...)`, build a confusion matrix for it on the held-out test set, and log the figure. After this it lives **next to the serialised model** in `./mlartifacts` — the same place `load_model` read from earlier. That is the artifact store made concrete: not just models, but any file you attach to a run.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Build a confusion matrix for the best (reloaded) model on the held-out test set.
fig, ax = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay.from_estimator(
    reloaded_model, X_test, y_test, ax=ax, colorbar=False
)
ax.set_title("Best run: confusion matrix (test set)")

# Reopen the best run and log the figure straight into its artifact store.
with mlflow.start_run(run_id=best_run_id):
    mlflow.log_figure(fig, "confusion_matrix.png")

plt.close(fig)

# Confirm the artifact now lives under the run, alongside the model.
artifacts = [a.path for a in client.list_artifacts(best_run_id)]
artifacts
```

**Where everything lives.** The two stores hold different things:

- **Backend store (`mlflow.db`, SQLite)** — the params, metrics, and **tags** we sorted and filtered on. `search_runs` queries this file.
- **Artifact store (`./mlartifacts` or `./mlruns`)** — the larger files: the serialised model that `load_model` read back, its signature and input example, and the **`confusion_matrix.png`** we just logged.

Start `mlflow server` against `sqlite:///mlflow.db` and you would see these runs in the web table, click the best one, read its tags, view the confusion-matrix image, and download the same model — the notebook and the UI are two views of one store.

## Pitfalls / when NOT to

MLflow is worth it the moment you run more than a couple of configurations — but it is not free, and a few habits keep it from biting:

- **Overkill for a throwaway one-off.** If you are fitting a single model once to eyeball a number, tracking it is ceremony. Reach for MLflow when runs start to accumulate and you need to compare them later.
- **Always use the `with mlflow.start_run()` context manager.** It guarantees the run **closes** even on error. A run left open by a bare `start_run()` will silently absorb the next run's logs.
- **Do not log giant artifacts every run.** A large model or a big plot logged on every iteration of a sweep can balloon the artifact store fast. Log heavy artifacts only for the runs that matter.
- **SQLite is single-user.** The SQLite backend here is perfect for one person on one laptop. A team uses `mlflow server` backed by **PostgreSQL** for the backend store and **S3** (or similar) for the artifact store, so many people log concurrently.

## Recap

- **MLflow records every run** — parameters, metrics, and the fitted model — so a search across configurations stays comparable and reproducible instead of scrolling off the screen.
- **Three storage layers:** backend store (params/metrics/tags, here SQLite), artifact store (models/files, here a folder), model registry (named versioned models).
- **`mlflow ui`** is a read-only viewer; **`mlflow server`** is the full server with REST API and serving.
- **The workflow:** `set_tracking_uri` + `set_experiment`, then one `with start_run():` per idea logging `log_params` / `log_metric` / `log_model`; compare with `search_runs`; reload with `load_model` and predict on raw columns.
- **Less typing, more notes:** `mlflow.autolog()` captures params/metrics/model for every `fit` with no manual calls; `set_tag` attaches searchable notes; `log_figure` puts a plot in the artifact store next to the model.

**Forward pointer:** the next mandatory notebook `lec_13d` adds `mlflow.evaluate`, validation gates, and the model registry with champion/challenger aliases. The optional career-track notebooks go further: `lec_13e` logs a sweep as parent/child runs with diagnostic figures, and `lec_13f` serves a registered model over REST.
