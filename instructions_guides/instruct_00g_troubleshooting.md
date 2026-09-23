# Guide 00g: troubleshooting

| Symptom                                           | Fix                                                                                                                                             |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `Cannot reach Ollama at http://localhost:11434`   | Start the Ollama app, or `ollama serve` in a terminal (guide 00b).                                                                              |
| `These models are not pulled yet`                 | Run the `ollama pull ...` lines the message prints.                                                                                             |
| `No MLflow server at http://127.0.0.1:5010`       | Start it from `mlflow_server/` (guide 00e).                                                                                                     |
| First answer takes 10-30 s                        | Normal: the model is loading into memory. Later calls are faster. `ollama ps` shows what is loaded.                                             |
| Answers are slow (< 5 tokens/s)                   | Another model may be loaded too (`ollama ps`); close other heavy apps; check `uvx llmfit recommend` (guide 00d); stay on `TIER = "cpu"`.        |
| The answer starts with `<think>` and is very long | Thinking mode is on. The notebooks pass `think=False` / `reasoning=False`; copy that argument.                                                  |
| `uv: command not found` after installing          | Reopen the terminal; on Windows check PATH (guide 00a).                                                                                         |
| PowerShell refuses to run the uv installer        | The command in guide 00a includes `-ExecutionPolicy ByPass`; run PowerShell as your user, not as administrator.                                 |
| `uv sync` fails on a package build                | Update uv (`uv self update`), delete `.venv/`, run `uv sync` again. Report the package name to the instructor.                                  |
| `ModuleNotFoundError: llm_course`                 | The notebook is not using `.venv/`: start JupyterLab with `uv run jupyter lab`, or in VS Code select the `.venv` kernel (guide 00a, section 4). |
| A cell about `HF_TOKEN` fails with 401            | The model is gated: accept its license on huggingface.co, or use an ungated model (guide 00c).                                                  |
| `Address already in use` when starting MLflow     | Another server holds the port. Use `--port 5011` and set `MLFLOW_TRACKING_URI` in `.env` (guide 00e).                                           |
| The Chroma folder seems corrupted or stale        | Stop the kernel and delete `data/chroma/`; the next notebook rebuilds it (2-4 minutes on CPU).                                                  |
| Greek text costs many more tokens than English    | Expected: tokenizers are trained mostly on English (lecture 1a).                                                                                |

## Where to look next

The official documentation of every tool in the course, with its source repository:

| Tool                                   | Documentation                                                                                                                                              | Source                                                                        |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| uv                                     | [docs.astral.sh/uv](https://docs.astral.sh/uv/)                                                                                                            | [astral-sh/uv](https://github.com/astral-sh/uv)                               |
| JupyterLab                             | [jupyterlab.readthedocs.io](https://jupyterlab.readthedocs.io/en/stable/)                                                                                  | [jupyterlab/jupyterlab](https://github.com/jupyterlab/jupyterlab)             |
| Ollama (application)                   | [docs.ollama.com](https://docs.ollama.com/)                                                                                                                | [ollama/ollama](https://github.com/ollama/ollama)                             |
| `ollama` (Python package)              | [pypi.org/project/ollama](https://pypi.org/project/ollama/)                                                                                                | [ollama/ollama-python](https://github.com/ollama/ollama-python)               |
| llmfit                                 | [llmfit.axjns.dev](https://llmfit.axjns.dev/)                                                                                                              | [AlexsJones/llmfit](https://github.com/AlexsJones/llmfit)                     |
| Hugging Face Hub and `huggingface_hub` | [huggingface.co/docs/hub](https://huggingface.co/docs/hub/index), [huggingface.co/docs/huggingface_hub](https://huggingface.co/docs/huggingface_hub/index) | [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) |
| transformers                           | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers/index)                                                                         | [huggingface/transformers](https://github.com/huggingface/transformers)       |
| LangChain                              | [docs.langchain.com](https://docs.langchain.com/)                                                                                                          | [langchain-ai/langchain](https://github.com/langchain-ai/langchain)           |
| LangGraph                              | [docs.langchain.com/oss/python/langgraph](https://docs.langchain.com/oss/python/langgraph/overview)                                                        | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)           |
| Chroma                                 | [docs.trychroma.com](https://docs.trychroma.com/)                                                                                                          | [chroma-core/chroma](https://github.com/chroma-core/chroma)                   |
| MLflow                                 | [mlflow.org/docs](https://mlflow.org/docs/latest/)                                                                                                         | [mlflow/mlflow](https://github.com/mlflow/mlflow)                             |
| Docker and Compose                     | [docs.docker.com](https://docs.docker.com/)                                                                                                                | [docker/compose](https://github.com/docker/compose)                           |
| pgvector                               | [README on GitHub](https://github.com/pgvector/pgvector)                                                                                                   | [pgvector/pgvector](https://github.com/pgvector/pgvector)                     |
