<!-- source: lectures_07_13_pandas_plots_scikit/lecture_13_pipelines_gridsearch_mlflow/reading_material/lec_13f_mlflow_model_serving.ipynb @ 0cc874704aaa -->

# Lecture 13f: Serving a Model over REST with MLflow

A trained, registered model is **useless until something can call it.** A web app, a mobile
backend, or another team's service does not import your Python code — it sends an HTTP request and
expects a prediction back. `mlflow models serve` turns a registered model into a live **REST
endpoint** that any application can `POST` to, with no glue code on your side.

**What this notebook covers:**

- Train a `Pipeline` and register it as `heart_disease_clf` with a **model signature**
  (`infer_signature`) and an `input_example` — the input/output schema the server enforces.
- A **guided demo** of the live server: `mlflow models serve`, a `curl` POST to `/invocations`,
  and the JSON response shape (shown as instructions, not run inside the notebook).
- The **headless equivalent**: `mlflow.pyfunc.load_model(...)` reproduces the exact REST
  request/response contract in-process, so you see it without a live server.
- **Signature enforcement**: what the server does when the request schema does not match.

> **Optional / career-track.** This notebook is deployment-track material. The mandatory core
> (`lec_13a`/`b`/`c`/`d`) already gets you tracking, pipelines, and the registry; serving is the last
> mile that ML / MLOps engineers own when a model goes to production. Covers goal **O3**.

```python
import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import requests
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.models import infer_signature

RANDOM_STATE = 42
```

## The data: heart-disease screening

We reuse the heart-disease dataset that runs through all of Lecture 13. Each row is a patient; the
target `Heart Disease` is `Presence` / `Absence`. We **sample 1500 rows** so everything stays fast —
the serving contract is identical at full size.

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y,
)
print("Train rows:", len(X_train), " Test rows:", len(X_test))
```

## Point MLflow at a local store

As in the other Lecture 13 MLflow notebooks, the executable cells log **directly** to a local SQLite
backend — no live server needed for *logging* and *registering*. The server only comes in later for
*serving*. We use a database and experiment name unique to this notebook so it does not collide with
the others.

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
mlflow.set_experiment("lecture13_serving")
print("Tracking URI:", mlflow.get_tracking_uri())
```

## Train the pipeline we will serve

We chain the `ColumnTransformer` (scale numeric, one-hot encode categorical) with a
`LogisticRegression` into a single `Pipeline`. Bundling preprocessing **inside** the model is what
makes serving clean: the endpoint receives **raw** patient columns and the pipeline scales and
encodes them internally, exactly as in training.

```python
preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
model.fit(X_train, y_train)
print("Test accuracy:", round(model.score(X_test, y_test), 3))
```

## The model signature: the schema the server enforces

A **signature** records the column names and dtypes the model expects as input, and the dtype it
returns as output. `infer_signature` reads this off real data: the training features and the model's
predictions. We pass it to `log_model`, and the serving endpoint then **enforces** it — a request
with the wrong columns or wrong types is rejected before it ever reaches the model.

The `input_example` is a few real rows stored alongside the model. It documents *exactly* what a
valid request looks like, and the UI uses it to pre-fill a sample payload.

```python
signature = infer_signature(X_train, model.predict(X_train))
input_example = X_train.iloc[:2]

print("Signature:")
print(signature)
```

## Log, register, and tag the model `champion`

We log the fitted pipeline with its signature and example, registering it under the name
`heart_disease_clf`. We then set the `champion` **alias** on the version we just created — a stable,
human-readable pointer (`models:/heart_disease_clf@champion`) that always resolves to "the version
currently in production", so the serving command never has to hard-code a version number.

```python
with mlflow.start_run(run_name="serve_candidate") as run:
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=input_example,
        registered_model_name="heart_disease_clf",
    )
print("Logged model URI:", info.model_uri)
```

```python
client = MlflowClient()
# The version we just registered is the newest one for this name.
latest = max(int(m.version) for m in client.search_model_versions("name='heart_disease_clf'"))
client.set_registered_model_alias("heart_disease_clf", "champion", version=latest)
print(f"Alias 'champion' -> heart_disease_clf version {latest}")
```

## Guided demo: serving over REST (run this in a terminal)

`mlflow models serve` starts a small web server that loads the model and exposes one endpoint,
`/invocations`. **This is a long-running process — it cannot run inside this notebook** (it would
block forever), so the commands below are for you to run in a **separate terminal**.

**1. Start the server** (point it at the aliased model, pick a free port):

```bash
mlflow models serve -m "models:/heart_disease_clf@champion" --port 5002 --no-conda
```

- `-m "models:/heart_disease_clf@champion"` — load the version the `champion` alias points to.
- `--port 5002` — listen on `http://127.0.0.1:5002`.
- `--no-conda` — use the current environment instead of building a fresh conda env.

**2. Send a prediction request** with `curl`. The body is a JSON `dataframe_split` object: a list
of `columns` and a list of `data` rows, one value per column, in order.

```bash
curl -X POST http://127.0.0.1:5002/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120", "EKG results", "Max HR", "Exercise angina", "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium"], "data": [[60, 1, 4, 130, 250, 0, 2, 150, 0, 1.5, 2, 0, 3]]}}'
```

**3. Read the response.** The server returns a JSON object with a `predictions` list — one entry per
input row:

```json
{"predictions": [0]}
```

**The `/invocations` contract:** every MLflow-served model speaks the same protocol — `POST` raw
feature columns as a `dataframe_split` (or `dataframe_records`) JSON body, get a `predictions` array
back. Any language that can make an HTTP request can call your model; nothing imports your Python.

## The headless equivalent: `pyfunc` reproduces the REST contract

We cannot run the live server here, but we do not need to in order to *see the contract*.
`mlflow.pyfunc.load_model` loads the registered model as a generic Python wrapper whose `.predict`
behaves **exactly** like the `/invocations` endpoint: hand it a DataFrame built from a `dataframe_split`
payload, get the same `predictions` back. This is the in-process twin of the `curl` call above.

```python
pyfunc_model = mlflow.pyfunc.load_model("models:/heart_disease_clf@champion")
print("Loaded:", type(pyfunc_model).__name__)
```

We build the **same `dataframe_split` JSON body** the REST client would send, from two real rows
of `X_test`. This is the literal request payload — `columns` plus `data` rows.

```python
sample_rows = X_test.iloc[:2]
# to_dict(orient="split") keeps each column's native type (ints stay ints, floats stay floats)
# -> exactly what a well-behaved REST client sends, and what the signature expects.
split_body = sample_rows.to_dict(orient="split")
split_body.pop("index")  # MLflow's dataframe_split only needs columns + data
payload = {"dataframe_split": split_body}
print(json.dumps(payload, indent=2))
```

Now we turn that JSON body back into a DataFrame (exactly what the server does on receipt) and call
`.predict`. The result is the `predictions` array the REST endpoint would put in its JSON response.

```python
split = payload["dataframe_split"]
request_df = pd.DataFrame(data=split["data"], columns=split["columns"])

predictions = pyfunc_model.predict(request_df)
print("REST response would be:", {"predictions": [int(p) for p in predictions]})
```

## Signature enforcement: bad requests are rejected

Because we logged a signature, the model knows which columns it requires. If a request is **missing a
column** (or sends the wrong type), schema enforcement raises an error *before* the model runs —
the live server turns this into an HTTP `400 Bad Request` instead of a misleading prediction. We
demonstrate by dropping a required column and catching the error.

```python
broken_df = request_df.drop(columns=["Cholesterol"])  # remove a required feature
try:
    pyfunc_model.predict(broken_df)
except Exception as err:
    msg = str(err)
    # The useful part is after "Error:" -> the schema mismatch reason.
    reason = msg.split("Error:")[-1].strip() if "Error:" in msg else msg
    print("Schema enforcement rejected the request.")
    print("Exception type:", type(err).__name__)
    print("Reason:", reason)
```

## Notes: from `serve` to production

`mlflow models serve` is perfect for local testing and small internal tools, but production serving
usually goes one step further:

- **Containerise it.** `mlflow models build-docker -m "models:/heart_disease_clf@champion" -n heart-clf`
  bakes the model and its `/invocations` server into a Docker image you can ship anywhere.
- **Deploy to a managed endpoint.** MLflow has deployment plugins for cloud targets (AWS SageMaker,
  Azure ML, Databricks, Kubernetes) that take the registered model URI and stand up a scalable,
  monitored endpoint — same `/invocations` contract, production-grade infrastructure underneath.

The contract you learned here — a signed model, a `dataframe_split` request, a `predictions`
response — is identical across all of these. That is the whole point of MLflow serving: **train and
register once, serve the same way everywhere.**
