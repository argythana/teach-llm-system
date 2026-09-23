#!/usr/bin/env python
"""Start the course MLflow server with the GitHub Copilot assistant provider available.

    cd mlflow_server
    uv run python ../tools/mlflow_server.py --host 127.0.0.1 --port 5010   # same flags as `mlflow server`

This is `mlflow server` with one change: uvicorn loads `llm_course.mlflow_app:app`, which
registers the Copilot provider (see llm_course/mlflow_copilot_provider.py) before
creating MLflow's FastAPI application. The Assistant panel then lists "GitHub Copilot CLI"
next to Claude Code, Codex CLI and Ollama, and uses it when it is the selected provider in
`~/.mlflow/assistant/config.json` (see tools/install_copilot_provider.py).

Students do not need this: the plain `mlflow server` command in the guides is enough for
every notebook. Run it from `mlflow_server/` so `mlflow.db` and `mlartifacts/` land there.
"""

import sys

import mlflow.server as mlflow_server

COURSE_APP = "llm_course.mlflow_app:app"

_original_build_uvicorn_command = mlflow_server._build_uvicorn_command


def _build_uvicorn_command_with_provider(
    uvicorn_opts, host, port, workers, app_name, env_file=None, is_factory=False
):
    return _original_build_uvicorn_command(
        uvicorn_opts, host, port, workers, COURSE_APP, env_file, is_factory
    )


def main():
    mlflow_server._build_uvicorn_command = _build_uvicorn_command_with_provider
    from mlflow.cli import cli

    args = sys.argv[1:] or ["--host", "127.0.0.1", "--port", "5010"]
    cli.main(args=["server", *args], prog_name="mlflow", standalone_mode=True)


if __name__ == "__main__":
    main()
