# Guide 00e: start the MLflow tracking server

From lecture 1d on, the notebooks record every model call as a **trace** in MLflow, the
tool you met in lecture 13 of the Python course. Before opening such a notebook, start
the server in a **separate terminal** from the `mlflow_server/` folder, so its database
and artifacts always land in one place. MLflow:
[source on GitHub](https://github.com/mlflow/mlflow),
[documentation](https://mlflow.org/docs/latest/).

Start it with:

```bash
cd mlflow_server
uv run mlflow server --host 127.0.0.1 --port 5010
```

Leave that terminal open and visit <http://127.0.0.1:5010>. The **Traces** tab of an
experiment is where the course's model calls appear.

The port is `5010`, not the `5000` you used in the Python course, because `5000` is
often taken (macOS AirPlay, other MLflow servers). If the notebook's check says
`No MLflow server at http://127.0.0.1:5010`, the server is not running.

## Where the data goes

`mlflow_server/mlflow.db` (a SQLite file) and `mlflow_server/mlartifacts/` are created
on first start. They are gitignored: they are your local history, not course material.
To start over, stop the server and delete those two.

## If you must use another port

Start the server with `--port <other>` and set
`MLFLOW_TRACKING_URI=http://127.0.0.1:<other>` in your `.env` file. The notebooks read
it.

## Windows

On Windows `mlflow server` uses the [`waitress`](https://github.com/Pylons/waitress) web
server, which `uv sync` installs there automatically. The command is the same.
