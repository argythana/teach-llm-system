# Guide 00b: install Ollama and pull the course models

**Ollama** runs open language models on your own computer and exposes them through a
local web server (`http://localhost:11434`), the same way `mlflow server` exposed
experiment tracking in the Python course. Nothing you type leaves your laptop.

Ollama is a **system-native application**: it installs like a browser or an editor, not
like a Python package, so `uv sync` does not install it. Source:
[GitHub](https://github.com/ollama/ollama); documentation:
[docs.ollama.com](https://docs.ollama.com/); the catalogue of models:
[ollama.com/library](https://ollama.com/library).

## 1. Install

- **Windows:** download the installer from <https://ollama.com/download> and run it.
  Ollama starts automatically and shows an icon in the system tray.
- **macOS:** download from <https://ollama.com/download>, or `brew install ollama`.
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`

Check in a terminal:

```bash
ollama --version
```

## 2. Pull the course models

Three models are used in the lectures. Downloads are one-time (about 2.2 GB in total):

```bash
ollama pull qwen3:1.7b          # the chat model (1.4 GB)
ollama pull qwen3:0.6b          # its smaller sibling, compared with it in lecture 1b (520 MB)
ollama pull nomic-embed-text    # the embedding model (274 MB), from lecture 2 on
```

If your laptop has a GPU with 8 GB of VRAM or more, you may also pull the GPU-tier model
used when you set `TIER = "gpu"` in a notebook:

```bash
ollama pull qwen3:8b            # 5.2 GB
```

Do not guess about the GPU tier: run `uvx llmfit` (guide 00d) to see what fits your
machine.

## 3. Check

```bash
ollama list          # the pulled models
ollama run qwen3:1.7b "Say hello in one sentence."
```

The first answer takes a few seconds while the model loads into memory. Type `/bye` to
leave an interactive `ollama run` session.

## The Ollama application and the `ollama` Python package

Two different things share the name:

|                 | The Ollama application                                                                                                                 | The `ollama` Python package                                                                                             |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| What it is      | a system-native program: the **server** on port `11434` plus the `ollama` command-line tool (`ollama pull`, `ollama run`, `ollama ps`) | a small Python **client library** (an SDK) that sends requests to that server                                           |
| Installed by    | the installer from ollama.com (this guide)                                                                                             | `uv sync`, from `pyproject.toml`                                                                                        |
| Runs the model? | yes, it loads the weights and generates tokens                                                                                         | no, it only talks to the application over HTTP                                                                          |
| Source          | [github.com/ollama/ollama](https://github.com/ollama/ollama)                                                                           | [github.com/ollama/ollama-python](https://github.com/ollama/ollama-python), [on PyPI](https://pypi.org/project/ollama/) |

The command line and the Python package talk to the same server: a model pulled with
`ollama pull` in a terminal is immediately available to `ollama.Client(...).chat(...)`
in a notebook. If the application is not running, the package can do nothing, which is
what the `Cannot reach Ollama` message means.

Where the course uses the Python package: lecture 1 talks to the server directly with it
(`lec_01a` to `lec_01c`: `client.chat`, `client.list`, `client.ps`, structured output
with `format=`), and `llm_course.checks.check_ollama` uses it in every configuration
cell. From `lec_01d` on, LangChain's `langchain-ollama` package talks to the same server
on your behalf, and in lecture 3 MLflow's judge does too; the application stays the one
thing that runs the models.

## Useful commands

| Command                                   | What it shows                                            |
| ----------------------------------------- | -------------------------------------------------------- |
| `ollama ps`                               | which models are loaded and whether on CPU or GPU        |
| `ollama show qwen3:1.7b`                  | context length, quantization, chat template              |
| `ollama rm <model>`                       | delete a model you no longer need                        |
| `ollama pull hf.co/<user>/<repo>:<quant>` | run a GGUF model straight from Hugging Face (lecture 1b) |

## If Ollama is not running

The notebooks' first cell raises `Cannot reach Ollama at http://localhost:11434`. Start
the Ollama app (Windows/macOS) or run `ollama serve` in a separate terminal (Linux).

## If Ollama runs somewhere else

Set `OLLAMA_HOST` in your `.env` file (see `.env.example`). The notebooks read it.
