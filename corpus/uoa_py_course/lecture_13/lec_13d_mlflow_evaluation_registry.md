<!-- source: lectures_07_13_pandas_plots_scikit/lecture_13_pipelines_gridsearch_mlflow/reading_material/lec_13d_mlflow_evaluation_registry.ipynb @ 0cc874704aaa -->

# Lecture 13d: Model Evaluation Gates and the Model Registry

Tracking (lec_13c) records *which* runs happened. The **model registry** is the next step: it turns a vetted run into a named, versioned, governed artefact.
The pattern this notebook teaches is the day-to-day reality of MLOps: promote one vetted model to **champion**, keep a **challenger** waiting in the wings, and **roll back** instantly if the champion regresses in production.

**What this notebook covers**

- `mlflow.evaluate` (static-dataset form) to compute a standard classifier metric report per model.
- A **custom business metric** + an explicit **validation gate** (an `if`) that decides whether a model is fit to register.
- Registering a passing model as a **named, versioned** entry in the model registry.
- **Champion / challenger aliases** via `MlflowClient`, reloading a model *by alias*, and **promotion + rollback**.

**Status: Mandatory reading.** This is the fourth mandatory notebook. Tracking (lec_13c) records runs; here you decide *whether a model is good enough* (evaluation gates) and *manage versions* (the registry, champion/challenger governance) — the step a data scientist meets the first time a model goes to production.

Forward pointer: lec_13f serves the registered `champion` model over REST with `mlflow models serve`.

```python
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

import requests
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

RANDOM_STATE = 42
```

## Point MLflow at a local SQLite store

The cell below logs directly to a SQLite backend store. This works headless — no live server needed — and is the *same* file a server would read.
To browse what we log here in the UI, start a server in a separate terminal:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
# then open http://127.0.0.1:5000
```

```python
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
mlflow.set_experiment("lecture13_eval_registry")
```

## Data: heart-disease, the running example

Same canonical contract as the rest of Lecture 13: a 1500-row sample of the heart-disease dataset, a binary target (`1 = Presence`, `0 = Absence`), and a fixed numeric/categorical column split.
We use one **stratified** train/test split and reuse it for *both* models, so the comparison is fair.

```python
df = (
    pd.read_csv("predict_heart_disease_train.csv")
    .drop(columns=["id"])
    .sample(n=1500, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

# Binary target: 1 = heart disease present, 0 = absent
y = (df["Heart Disease"] == "Presence").astype(int)
X = df.drop(columns=["Heart Disease"])

NUMERIC = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
CATEGORICAL = ["Sex", "Chest pain type", "FBS over 120", "EKG results",
               "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
X_train.shape, X_test.shape
```

## Two pipeline models on the same split

`preprocess` scales the numeric columns and one-hot encodes the categorical ones.
We wrap it with a classifier inside a `Pipeline`, so preprocessing is fit on the *training* data only — no leakage.
We build two candidates:

- **`champion_candidate`** — `LogisticRegression`, a calibrated, interpretable linear baseline.
- **`challenger`** — `RandomForestClassifier`, a strong default tree-ensemble baseline.

```python
preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])
```

```python
champion_candidate = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
champion_candidate.fit(X_train, y_train)

challenger = Pipeline([
    ("preprocess", preprocess),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
])
challenger.fit(X_train, y_train)
print("Both pipelines fitted.")
```

## Evaluate each model with `mlflow.evaluate` (static-dataset form)

`mlflow.evaluate` can compute a full classifier metric report from a **dataframe of predictions** — no model object or URI required.
We hand it three columns: the features (carried along), the true `label`, and the model's `prediction`.

We wrap each evaluation in its own MLflow run, so the metrics land in the tracking store next to the model we will register.
We also build a **confusion-matrix figure** and log it with `mlflow.log_figure` *inside the same run* — so the evaluation evidence travels with the run as an artifact.

- **Metrics** answer "how good?" as numbers — accuracy, recall, the confusion counts.
- **The figure** answers the same question visually, and is the kind of audit evidence a registered model should carry.

The helper below returns the metrics dict so we can feed it into the gate.

```python
def evaluate_model(model, run_name, fig_label):
    
    # Log an mlflow.evaluate report for `model` on the test set; return result.metrics.
    predictions = model.predict(X_test)
    eval_df = X_test.copy()
    eval_df["label"] = y_test.values
    eval_df["prediction"] = predictions
    
    
    with mlflow.start_run(run_name=run_name):
        result = mlflow.evaluate(
            data=eval_df,
            predictions="prediction",
            targets="label",
            model_type="classifier",
        )
        
        # Build a confusion-matrix figure and log it as an artifact inside this run.
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, display_labels=["Absence", "Presence"], ax=ax
        )
        ax.set_title(f"Confusion matrix: {fig_label}")
        mlflow.log_figure(fig, f"confusion_matrix_{fig_label}.png")
        plt.close(fig)
    return result.metrics
```

```python
champion_metrics = evaluate_model(champion_candidate, "eval_logreg", "logreg")

# Show a few headline metrics from the returned report
keys = ["accuracy_score", "f1_score",
        "true_positives", "false_positives",
        "true_negatives", "false_negatives"]
{k: champion_metrics[k] for k in keys}
```

```python
challenger_metrics = evaluate_model(challenger, "eval_randomforest", "randomforest")
{k: challenger_metrics[k] for k in keys}
```

## A custom business metric + a validation gate

Accuracy alone hides a clinical cost: **missing a sick patient** (a false negative) is far worse than a false alarm.
So we define a business metric — **recall on the `Presence` class** — straight from the confusion counts:

$$\text{recall} = \frac{\text{true positives}}{\text{true positives} + \text{false negatives}}$$

The **gate** is a plain `if`: we only register a model if it clears *both* bars.

- `accuracy >= 0.80` — it must be broadly correct.
- `recall >= 0.75` — it must catch most genuinely sick patients.

```python
ACCURACY_BAR = 0.80
RECALL_BAR = 0.75

def passes_gate(metrics):
    # Custom business gate: high accuracy AND high recall on the Presence class.
    accuracy = metrics["accuracy_score"]
    tp = metrics["true_positives"]
    fn = metrics["false_negatives"]
    recall = tp / (tp + fn)
    ok = (accuracy >= ACCURACY_BAR) and (recall >= RECALL_BAR)
    return ok, accuracy, recall
```

```python
champ_ok, champ_acc, champ_rec = passes_gate(champion_metrics)
chal_ok, chal_acc, chal_rec = passes_gate(challenger_metrics)

print(f"LogReg       accuracy={champ_acc:.3f} recall={champ_rec:.3f} -> gate {'PASS' if champ_ok else 'FAIL'}")
print(f"RandomForest accuracy={chal_acc:.3f} recall={chal_rec:.3f} -> gate {'PASS' if chal_ok else 'FAIL'}")
```

## Register the models that pass the gate

`mlflow.sklearn.log_model(..., registered_model_name=...)` does two things at once: it logs the fitted pipeline as a run artefact **and** creates a new **version** under a named registry entry.

- A **signature** (`infer_signature`) records the expected input/output schema — the registry enforces it later.
- An **`input_example`** stores a few real rows so the UI can show what a request looks like.

We register the LogReg model first (becomes **version 1**), then the RandomForest (**version 2**).
We only call `log_model` inside the gate, so a failing model never reaches the registry.

```python
REGISTERED_NAME = "heart_disease_clf"
signature = infer_signature(X_train, champion_candidate.predict(X_train))

def register_if_passes(model, gate_ok, run_name):
    if not gate_ok:
        print(f"{run_name}: gate FAILED — not registered.")
        return None
    with mlflow.start_run(run_name=run_name):
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.iloc[:3],
            registered_model_name=REGISTERED_NAME,
        )
    return info
```

```python
# Version 1: the LogReg champion candidate
register_if_passes(champion_candidate, champ_ok, "register_logreg")
# Version 2: the RandomForest challenger
register_if_passes(challenger, chal_ok, "register_randomforest")

client = MlflowClient()
versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
sorted((int(v.version), v.run_id[:8]) for v in versions)
```

## Champion / challenger aliases

A registry **alias** is a *mutable, named pointer* to one immutable version.
We point `champion` at the vetted version 1 and `challenger` at version 2, then reload **by alias** and predict.
Code that loads `models:/heart_disease_clf@champion` never needs to know which numeric version is live — that is the whole point.

```python
client.set_registered_model_alias(REGISTERED_NAME, "champion", version=1)
client.set_registered_model_alias(REGISTERED_NAME, "challenger", version=2)

champion_model = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")
preds = champion_model.predict(X_test.iloc[:5])
print("champion -> version 1, sample predictions:", preds.tolist())
```

## See the registry in the MLflow UI

Open **http://127.0.0.1:5000** and go to **Models -> `heart_disease_clf`**. You will see:

- the **versions** you registered (v1, v2),
- the **`@champion`** and **`@challenger`** aliases moving as you promote and roll back,
- each version's run, metrics, and the **confusion-matrix artifact** logged during evaluation.

This is the same governed view an MLOps team uses to decide what is live in production.

## Promotion and rollback

Promotion is just *repointing an alias*. Suppose the challenger (version 2) wins a head-to-head — we promote it by aiming the `champion` alias at version 2.
If version 2 then misbehaves in production, **rollback** is equally cheap: aim `champion` back at version 1.

- **Aliases are mutable** — repoint them freely; this is the live/rollback switch.
- **Versions are immutable** — version 1 always means the exact LogReg artefact we registered; promoting never rewrites history.

```python
# Promote the challenger: champion now points at version 2
client.set_registered_model_alias(REGISTERED_NAME, "champion", version=2)
promoted = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")
print("after promotion, champion -> version 2:", promoted.predict(X_test.iloc[:5]).tolist())
```

```python
# Rollback: version 2 regressed — aim champion back at version 1
client.set_registered_model_alias(REGISTERED_NAME, "champion", version=1)
rolled_back = client.get_model_version_by_alias(REGISTERED_NAME, "champion")
print(f"after rollback, champion -> version {rolled_back.version}")
```

## Notes

- **The registry is your audit trail.** It records which version was the `champion` when — the governance answer to "what model was live in production on date X?".
- **Gate before you register.** A custom business metric (here: recall on sick patients) plus an explicit `if` keeps unfit models out of the registry entirely.
- **Aliases vs versions.** Versions are immutable artefacts; aliases (`champion`, `challenger`) are mutable pointers. Promotion and rollback are one alias-repoint each — fast, reversible, and fully logged.
- **`mlflow.evaluate` static form** gives a standard metric report from a predictions dataframe, with no model object or URI — robust and headless-safe.
