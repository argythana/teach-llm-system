<!-- source: lectures_07_13_pandas_plots_scikit/lecture_13_pipelines_gridsearch_mlflow/reading_material/lec_13g_automl_autogluon.ipynb @ 0cc874704aaa -->

# Lecture 13g (optional, experimental): AutoML with AutoGluon

> **Status: optional / experimental.** This notebook is **not** part of the required
> Lecture 13 material and is intentionally left out of `goals_13.md`. It exists to show
> what an *AutoML* framework does, by re-solving the **exact same heart-disease problem**
> you built up by hand in `lec_13a`–`lec_13d` — but in a handful of lines.

Across Lecture 13 you assembled an ML automation workflow piece by piece:

| Notebook | What you built by hand |
| --- | --- |
| `lec_13a` | `GridSearchCV` / `RandomizedSearchCV` — automated **hyperparameter tuning** |
| `lec_13b` | A leakage-free `Pipeline` + `ColumnTransformer` — **preprocessing** + **multi-model search** |
| `lec_13b` (archive: stacking) | Comparing model **families** and combining them by **stacking** |
| `lec_13c`–`lec_13d` | **MLflow** tracking, evaluation, and the model registry |

**AutoML** asks: *can a single library do all of that for you?* [AutoGluon](https://auto.gluon.ai/)
(by AWS) is one answer. Given a raw `DataFrame` and the name of the target column, it:

- infers feature types and **preprocesses** automatically (impute, encode, scale) — no `ColumnTransformer`;
- trains **many model families** (LightGBM, XGBoost, CatBoost, random forests, extra-trees, k-NN, linear, neural nets);
- **tunes** them and combines the best by **multi-layer stacking + bagging**;
- ranks everything on a **leaderboard** for the **metric you choose**.

The point of this notebook is **contrast**, not replacement. Knowing what each line *does* —
the thing Lectures 13a–13d taught you — is exactly what lets you trust, debug, and deploy an
AutoML result instead of treating it as a black box.

## Setup: installing AutoGluon (not in the course `requirements.txt`)

AutoGluon is a **heavy** dependency (it pulls in LightGBM, XGBoost, CatBoost, and
PyTorch). It is deliberately **left out** of the course environment so the student
install stays light. To run this notebook, install it into your venv once:

```bash
pip install "autogluon.tabular[lightgbm,catboost,xgboost]"
```

(The `[...]` extras pull in the gradient-boosting models AutoGluon leans on. The bare
`autogluon.tabular` still works — it just trains fewer model families.)

```python
# Run this once if AutoGluon is not yet installed (uncomment):
# %pip install "autogluon.tabular[lightgbm,catboost,xgboost]"
```

```python
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor

RANDOM_STATE = 42
pd.set_option("display.max_columns", None)
```

## 1. The same data, the same target — but handed over **raw**

We load the identical heart-disease dataset used throughout Lecture 13 and take the same
1500-row teaching sample. The crucial difference from `lec_13b`: we do **not** split the
columns into `NUMERIC` / `CATEGORICAL`, build a `ColumnTransformer`, scale, or one-hot
encode anything. AutoGluon wants the **raw frame with the label column still in it** and
figures the rest out itself.

```python
# Heart-disease dataset (committed alongside this notebook). 630k rows -> sample for fast teaching.
df = (
    pd.read_csv("predict_heart_disease_train.csv")
    .drop(columns=["id"])
    .sample(n=1500, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

LABEL = "Heart Disease"          # AutoGluon predicts this column directly ("Presence"/"Absence")

# AutoGluon splits features from the label itself, so we keep the label IN the frame and
# only split rows into train/test. No ColumnTransformer, no manual encoding.
train_df, test_df = train_test_split(
    df, test_size=0.25, random_state=RANDOM_STATE, stratify=df[LABEL]
)

train_data = TabularDataset(train_df)   # a thin pandas subclass AutoGluon understands
test_data = TabularDataset(test_df)

print("Train rows:", len(train_data), " Test rows:", len(test_data))
print("\nClass balance (train):")
print(train_data[LABEL].value_counts(normalize=True).round(3))
train_data.head(3)
```

## 2. One `.fit()` replaces the whole pipeline

In `lec_13b` a single `GridSearchCV` over a pipeline compared two classifier families. Here,
one `TabularPredictor(...).fit(...)` call does the preprocessing, trains **a dozen models
across many families**, tunes them, and stacks the winners.

A few arguments worth knowing:

- `label` — the column to predict.
- `eval_metric` — the metric models are **selected and ranked** by (the AutoML echo of
  "choosing the scoring metric" in `lec_13a` / `lec_13c`). We start with `"accuracy"`.
- `path` — where the trained models are written (a folder; gitignored here).
- `time_limit` — a wall-clock budget. AutoML's honesty knob: it trains until the clock runs
  out, then stops. 120s is plenty for 1500 rows.
- `verbosity=1` — keep the log short for teaching (the default `2` is very chatty).

```python
predictor = TabularPredictor(
    label=LABEL,
    eval_metric="accuracy",
    path="ag_models/accuracy",          # gitignored; see .gitignore
    verbosity=1,
)

predictor.fit(
    train_data,
    time_limit=120,                     # wall-clock budget in seconds
    presets="medium_quality",           # fast, single-layer; "best_quality" (used in S4) stacks + bags
)

print("\nProblem type AutoGluon inferred:", predictor.problem_type)
print("Optimised for metric:", predictor.eval_metric.name)
print("Model picked to predict with:", predictor.model_best)
```

## 3. The leaderboard = model selection across families, in one table

This is the AutoML counterpart to `lec_13b` §7 ("comparing classifier families in ONE
search") and the archived stacking notebook's family-by-family bake-off. `leaderboard(test_data)`
fits **nothing new** — it just scores every already-trained model on the held-out test set and
ranks them.

Read the columns like a `cv_results_` table:

- `model` — the algorithm (note the `WeightedEnsemble_L2` row: a **stacked** combination, see §4);
- `score_test` — test-set score on our `eval_metric` (accuracy);
- `score_val` — the internal validation score AutoGluon **selected** on (it never saw the test set);
- `fit_time` / `pred_time_test` — the cost trade-off, just like `lec_13a`'s timing table.

```python
leaderboard = predictor.leaderboard(test_data, silent=True)
leaderboard[["model", "score_test", "score_val", "fit_time", "pred_time_test"]].round(4)
```

## 4. Multi-layer stacking + bagging — the archived stacking notebook, automated

The archived `lec_13b` stacking notebook had you hand-build a `StackingClassifier`: train
base learners, then a meta-learner on their out-of-fold predictions. AutoGluon does exactly
this, automatically, when you ask for the `best_quality` preset:

- **Bagging** — each model is trained on several cross-validation folds and averaged (the
  leakage-free out-of-fold idea from `lec_13b` §5), which is what feeds an honest stack.
- **Multi-layer stacking** — base models form **layer 1** (`_L1`); their out-of-fold
  predictions become features for **layer 2** (`_L2`) models; a final weighted ensemble
  blends them.

The model names carry the stack level: `LightGBM_BAG_L1`, `..._L2`, `WeightedEnsemble_L2/L3`.

```python
stack_predictor = TabularPredictor(
    label=LABEL,
    eval_metric="accuracy",
    path="ag_models/stack",
    verbosity=1,
).fit(
    train_data,
    time_limit=180,
    presets="best_quality",             # turns ON bagging + multi-layer stacking
)

stack_lb = stack_predictor.leaderboard(test_data, silent=True)
# The "stack_level" column shows which layer each model lives on.
stack_lb[["model", "score_test", "score_val", "stack_level", "fit_time"]].round(4)
```

```python
# Confirm the best model really is a multi-layer stacked ensemble, and see what feeds it.
best = stack_predictor.model_best
print("Best model:", best, "(stack level", stack_lb.set_index("model").loc[best, "stack_level"], ")")
print("\nModels feeding the final ensemble:")
info = stack_predictor.info()
base = info["model_info"][best].get("stacker_info", {}).get("base_model_names")
print(base if base else "(weighted ensemble over all L1/L2 models)")
```

## 5. Choosing the right metric — `lec_13c`, in one argument

`lec_13a` and the archived `lec_13c` metric notebook made the case that **accuracy is the
wrong target** when one error type costs more than the other — for heart-disease screening, a
missed case (false negative) is far worse than a false alarm. There you switched
`GridSearchCV(scoring="recall")`; here you switch a single `eval_metric`.

We re-fit optimising for **`roc_auc`** (ranking quality across all thresholds) and watch the
leaderboard order — and the model AutoGluon *selects* — change.

```python
auc_predictor = TabularPredictor(
    label=LABEL,
    eval_metric="roc_auc",              # <-- the only change vs. S2
    path="ag_models/roc_auc",
    verbosity=1,
).fit(train_data, time_limit=120, presets="medium_quality")

auc_lb = auc_predictor.leaderboard(test_data, silent=True)
print("Optimised for:", auc_predictor.eval_metric.name)
auc_lb[["model", "score_test", "score_val"]].round(4).head(8)
```

```python
# Side-by-side: the accuracy-optimised winner vs. the roc_auc-optimised winner can differ,
# exactly as the accuracy-vs-recall winners differed in lec_13a.
print(f"accuracy-optimised best model : {predictor.model_best}")
print(f"roc_auc-optimised  best model : {auc_predictor.model_best}")
```

## 6. Predicting on brand-new raw patients — `lec_13b` §7d, no preprocessing needed

In `lec_13b` you fed new patients through `best_pipeline.predict(...)` and the pipeline
re-applied the fitted scaler/encoder. AutoGluon is the same idea taken further: hand it a
**raw** frame with the original column names and it re-applies every learned transformation
internally. Note we pass the columns exactly as they appear in the CSV.

```python
# Two brand-new raw patients, in the ORIGINAL column format (no scaling/encoding by us).
new_patients = pd.DataFrame([
    {"Age": 67, "Sex": 1, "Chest pain type": 4, "BP": 160, "Cholesterol": 286,
     "FBS over 120": 0, "EKG results": 2, "Max HR": 108, "Exercise angina": 1,
     "ST depression": 1.5, "Slope of ST": 2, "Number of vessels fluro": 3, "Thallium": 7},
    {"Age": 41, "Sex": 0, "Chest pain type": 2, "BP": 112, "Cholesterol": 204,
     "FBS over 120": 0, "EKG results": 0, "Max HR": 172, "Exercise angina": 0,
     "ST depression": 0.0, "Slope of ST": 1, "Number of vessels fluro": 0, "Thallium": 3},
])

pred = predictor.predict(new_patients)            # uses the accuracy-optimised predictor
proba = predictor.predict_proba(new_patients)

out = new_patients[["Age", "Sex", "BP", "Cholesterol", "Max HR"]].copy()
out["prediction"] = pred.values
out["P(Presence)"] = proba["Presence"].round(3).values
out
```

## 7. Looking inside the black box: feature importance

`lec_13b` §8 had you reach into a fitted pipeline to read engineered feature names — the
discipline of *not* trusting a model blindly. AutoGluon's `feature_importance()` is the same
instinct: it permutes each column on the test set and measures the drop in score, so you can
sanity-check that the model leans on **clinically plausible** signals (chest-pain type,
thallium scan, vessels) rather than noise.

```python
importance = predictor.feature_importance(test_data)
importance[["importance", "stddev", "p_value"]].round(4)
```

## 8. Hyperparameter tuning — `lec_13a`, delegated

`lec_13a` was entirely about searching a hyperparameter grid by hand. AutoGluon exposes the
same idea two ways:

1. **Presets** (`medium_quality`, `good_quality`, `best_quality`) — the lazy, robust default.
   Each preset is a curated bundle of models *and* search effort. This is the knob you'll
   reach for 95% of the time.
2. **Explicit HPO** via `hyperparameter_tune_kwargs` — a `RandomizedSearchCV`-style budget
   over a search space you define per model, for when you want fine control.

Below we run a small explicit search over LightGBM only, to keep it fast and legible.

```python
from autogluon.common import space  # AutoGluon's search-space objects (like scipy distributions)

hpo_predictor = TabularPredictor(
    label=LABEL, eval_metric="accuracy", path="ag_models/hpo", verbosity=1,
).fit(
    train_data,
    time_limit=120,
    # Search space for ONE model family, the way you'd define a grid in lec_13a:
    hyperparameters={
        "GBM": {
            "num_boost_round": 200,
            "learning_rate": space.Real(0.01, 0.3, log=True),
            "num_leaves": space.Int(16, 96),
        }
    },
    hyperparameter_tune_kwargs={"num_trials": 8, "scheduler": "local", "searcher": "random"},
)

hpo_lb = hpo_predictor.leaderboard(test_data, silent=True)
print("Trials explored, ranked by validation accuracy:")
hpo_lb[["model", "score_test", "score_val"]].round(4)
```

## 9. Tracking an AutoML run with MLflow — closing the loop with `lec_13c`

AutoGluon has no MLflow `autolog`, but the tracking discipline from `lec_13c` applies
unchanged: log the **params** (metric, preset, time budget), the **metrics** (best
validation/test score), and the **artifacts** (the leaderboard, the trained predictor folder).
We reuse the exact SQLite-fallback pattern from `lec_13c` so the run shows up in the same
MLflow UI alongside your hand-built pipelines.

```python
import mlflow, requests

MLFLOW_UI = "http://127.0.0.1:5000"

def mlflow_server_running(uri=MLFLOW_UI, timeout=2):
    try:
        return requests.get(f"{uri}/health", timeout=timeout).status_code == 200
    except requests.exceptions.RequestException:
        return False

if mlflow_server_running():
    mlflow.set_tracking_uri(MLFLOW_UI)
    print(f"MLflow server is UP at {MLFLOW_UI} -- runs appear live in the web UI.")
else:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    print("MLflow server is NOT running -- logging to local SQLite store 'mlflow.db'.")
mlflow.set_experiment("lecture13_automl")
```

```python
best_row = leaderboard.sort_values("score_test", ascending=False).iloc[0]

with mlflow.start_run(run_name="autogluon_accuracy"):
    # Params: the knobs that defined this AutoML run.
    mlflow.log_params({
        "framework": "autogluon",
        "eval_metric": predictor.eval_metric.name,
        "preset": "medium_quality",
        "time_limit_s": 120,
        "best_model": best_row["model"],
    })
    # Metrics: what the run achieved.
    mlflow.log_metrics({
        "best_score_test": float(best_row["score_test"]),
        "best_score_val": float(best_row["score_val"]),
        "n_models_trained": int(len(leaderboard)),
    })
    # Artifacts: the full leaderboard as a CSV, so the comparison is reproducible.
    leaderboard.to_csv("ag_leaderboard.csv", index=False)
    mlflow.log_artifact("ag_leaderboard.csv")

print("Logged AutoGluon run to MLflow.")
print("Inspect it next to your lec_13c/13d runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
```

## Pitfalls / when NOT to reach for AutoML

AutoGluon is genuinely strong on tabular data, but the Lecture 13 fundamentals are what keep
you out of trouble:

- **Leakage is still your job.** AutoGluon prevents leakage *across its internal folds*, but
  if **you** leak — passing a column computed from the label, or letting the test set into
  `fit()` — it will happily learn the leak. The `lec_13b` leakage lesson does not go away.
- **It is a black box by default.** A `WeightedEnsemble_L2` over eight bagged models is hard
  to explain to a clinician or an auditor. When you need an interpretable model, the hand-built
  `LogisticRegression` from `lec_12`/`lec_13b` may win on *trust* even if it loses on accuracy.
- **Compute and size.** `best_quality` trains dozens of models and writes hundreds of MB to
  disk. That is fine for a laptop experiment, expensive for a deployment. `mlflow models serve`
  (`lec_13f`) of a giant ensemble is a real cost.
- **Metric choice is still a human decision.** AutoGluon optimises whatever `eval_metric` you
  give it — it cannot know that a false negative is the costly error here. §5 only works because
  *you* chose `roc_auc`/recall, exactly as in `lec_13c`.
- **Don't skip the fundamentals.** AutoML is fastest in the hands of someone who could build the
  pipeline by hand — because they can read the leaderboard, smell a leak, and pick the metric.

## Recap: every Lecture 13 technique → its AutoGluon one-liner

| Lecture 13 (by hand) | AutoGluon (this notebook) |
| --- | --- |
| `ColumnTransformer` + scale/encode/impute (`13b`) | automatic — pass the raw frame (§1–2) |
| `Pipeline` chaining preprocessing + model (`13b`) | inside `TabularPredictor.fit()` (§2) |
| Comparing classifier families in one search (`13b`) | `predictor.leaderboard()` (§3) |
| Hand-built `StackingClassifier` (`13b` archive) | `presets="best_quality"` → `_L1`/`_L2` stack (§4) |
| `GridSearchCV(scoring=...)` metric choice (`13a`/`13c`) | `eval_metric=...` (§5) |
| `RandomizedSearchCV` budget (`13a`) | `hyperparameter_tune_kwargs` / presets (§8) |
| `pipeline.predict(new_raw_rows)` (`13b` §7d) | `predictor.predict(raw_df)` (§6) |
| Inspecting the fitted pipeline (`13b` §8) | `predictor.feature_importance()` (§7) |
| MLflow params/metrics/artifacts (`13c`) | same MLflow API, AutoGluon as the model (§9) |

**The takeaway:** AutoML compresses the *mechanics* of Lecture 13 into a few calls — but every
*judgement* it relies on (which metric, is there a leak, is the winner trustworthy enough to
ship) is exactly what Lectures 13a–13d taught you to make. The library is the easy part.
