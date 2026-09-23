"""Environment checks that fail early with a friendly message.

Written in lec_01a. A cryptic connection error two cells later is the most common
way a beginner loses ten minutes; these checks name the fix instead.
"""

GUIDES = "instructions_guides"


def check_ollama(host, models):
    """Raise RuntimeError unless Ollama answers at ``host`` and every model is pulled.

    ``models`` is a list of tags such as ``["qwen3:1.7b", "nomic-embed-text"]``.
    """
    import ollama

    client = ollama.Client(host=host)
    try:
        installed = {m.model for m in client.list().models}
    except Exception as exc:  # ConnectionError, httpx errors, ...
        raise RuntimeError(
            f"Cannot reach Ollama at {host}.\n"
            "Is it running? Start the Ollama app, or run `ollama serve` in a terminal.\n"
            f"Install guide: {GUIDES}/instruct_00b_install_ollama.md\n"
            f"(original error: {exc})"
        ) from None

    # "qwen3:1.7b" is listed as "qwen3:1.7b"; a bare "nomic-embed-text" as "nomic-embed-text:latest".
    def is_pulled(tag):
        return tag in installed or f"{tag}:latest" in installed

    missing = [m for m in models if not is_pulled(m)]
    if missing:
        pulls = "\n".join(f"  ollama pull {m}" for m in missing)
        raise RuntimeError(
            "These models are not pulled yet. Run in a terminal:\n"
            f"{pulls}\n"
            f"then re-run this cell. Guide: {GUIDES}/instruct_00b_install_ollama.md"
        )
    print(f"Ollama OK at {host}; models ready: {', '.join(models)}")


def check_mlflow(tracking_uri):
    """Raise RuntimeError unless an MLflow tracking server answers at ``tracking_uri``."""
    import requests

    try:
        response = requests.get(f"{tracking_uri.rstrip('/')}/health", timeout=5)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"No MLflow server at {tracking_uri}.\n"
            "Start it in a separate terminal, from the mlflow_server/ folder:\n"
            "  cd mlflow_server\n"
            "  uv run mlflow server --host 127.0.0.1 --port 5010\n"
            f"Guide: {GUIDES}/instruct_00e_start_mlflow_server.md\n"
            f"(original error: {exc})"
        ) from None
    print(f"MLflow OK at {tracking_uri}")
