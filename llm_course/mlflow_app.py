"""The MLflow server application with the GitHub Copilot assistant provider registered.

`tools/mlflow_server.py` starts uvicorn on this module instead of MLflow's own
`mlflow.server.fastapi_app:app`, so the provider is registered in the server process
before the Assistant API enumerates providers. Nothing else differs from `mlflow server`.
"""

from llm_course.mlflow_copilot_provider import install

install()

from mlflow.server.fastapi_app import app  # noqa: E402  (must come after install())

__all__ = ["app"]
