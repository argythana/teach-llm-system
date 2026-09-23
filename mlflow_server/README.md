# MLflow server folder

Start the course tracking server from **this folder** so its SQLite database and
artifacts land here:

```bash
cd mlflow_server
uv run mlflow server --host 127.0.0.1 --port 5010
```

Open <http://127.0.0.1:5010>. See
`instructions_guides/instruct_00e_start_mlflow_server.md`. The files it creates
(`mlflow.db`, `mlartifacts/`) are gitignored: they are your local history, not course
material.

## What the server holds

| Experiment                                                        | Written by                       | What you see                                                                                 |
| ----------------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------- |
| `llm-course-01-basics`, `llm-course-02-rag`, `llm-course-03-eval` | the lecture notebooks            | traces of every model call (lecture 1d on), evaluation runs (lecture 3)                      |
| `llm-course-01-exercises` and siblings                            | the exercise notebooks           | the same, for the practice exercises                                                         |
| `ai-collaboration-log`                                            | `tools/log_session_to_mlflow.py` | the sessions in which this course was built with an AI assistant, ported into MLflow (below) |

## The course's own build, ported into MLflow

This course was designed and written with Claude Code (an AI coding assistant running
Claude models). The transcripts are archived in `ai_collaboration_log/`, and the same
tooling lecture 3 teaches is applied to them:

- **Traces.** Each instructor turn becomes one MLflow trace: the instructor's message as
  input, the assistant's reply as output, and one child span per tool call.

- **Judges.** Three independent judges score the replies against
  `ai_collaboration_log/evaluation/rubric.md`, and the Traces tab shows them side by
  side:

  - a **blind Claude judge**: a fresh Claude Code sub-agent on a Sonnet model that never
    saw the session (`judge/*` feedback, source `LLM_JUDGE`);
  - the **local judge** the course uses everywhere: `mlflow.genai.evaluate` with
    deterministic scorers and `Guidelines` scorers run by the Ollama model
    (`ollama:/qwen3:1.7b` on the `cpu` tier);
  - optionally a **GitHub Copilot judge** from a different model family
    (`tools/judge_with_copilot.py`, default model `gpt-5.4`), for anyone with a Copilot
    subscription, which students and educators get through GitHub Education.

  Why three: a model judging its own family can be lenient, a small local model is weak
  on long technical replies, and a second family is the cheapest independent opinion.
  The disagreements are the interesting part.
  `ai_collaboration_log/evaluation/README.md` has the procedure and the caveats.

## The MLflow Assistant (experimental): Copilot by default on the instructor's machine

MLflow 3.16 ships an experimental **Assistant**: a chat panel in the UI that can analyse
traces and evaluation runs, backed by a coding-agent CLI, an API key, or an
OpenAI-compatible server. Providers in this version: Claude Code, OpenAI Codex CLI,
Ollama, the API providers (OpenAI, Anthropic, Gemini), any OpenAI-compatible endpoint,
and the MLflow AI Gateway. When no provider has been chosen, MLflow uses the first CLI
it finds installed, Claude Code before Codex, which is why the panel first opened on
Claude on this machine without any configuration.

This repository adds a **GitHub Copilot CLI** provider
(`llm_course/mlflow_copilot_provider.py`), built like MLflow's own Claude Code provider:
one non-interactive `copilot -p` run per turn, resumed on the next turn, with MLflow's
skills available to it. It uses only the official CLI and the instructor's Copilot
subscription (GitHub Education), one premium request per turn, and never the Azure
OpenAI deployment that the instructor's Codex is configured with. Two commands wire it
in:

```bash
uv run python tools/install_copilot_provider.py         # once: ~/.mlflow/assistant/config.json (Copilot selected,
                                                        # model gpt-5.4; Claude Code, Codex, Ollama kept) and
                                                        # MLflow's skills copied to ~/.copilot/skills/
cd mlflow_server && uv run python ../tools/mlflow_server.py --host 127.0.0.1 --port 5010
```

The launcher is `mlflow server` with one change: uvicorn loads `llm_course.mlflow_app`,
which registers the provider before the Assistant API enumerates providers. A plain
`mlflow server` still works for everything else and shows the built-in providers only.
The other providers stay in the menu; switch there, or edit
`~/.mlflow/assistant/config.json` (`selected`, `model`).
`tools/install_copilot_provider.py --remove` undoes the config entry.

## Notes from exploring GitHub-provided models (September 2026)

- **GitHub Models**, the free OpenAI-compatible inference endpoint that came with a
  GitHub account, was retired on 30 July 2026; its API now returns a retirement notice.
  It would have been the easiest non-Claude judge for students.
- **GitHub Copilot CLI** (`npm install -g @github/copilot`, Node 22+) and the **Copilot
  SDK** (`github-copilot-sdk` on PyPI) run prompts non-interactively with your GitHub
  login (`copilot -p "..." --model gpt-5.4 --output-format json`) and count each prompt
  against the plan's AI credits. Its default model on this plan was `claude-sonnet-5`,
  so a "different family" judge must name a model explicitly; `gpt-5.4` worked. Every
  call carries the CLI's large system prompt (about 25k tokens), so it is fine for a few
  dozen judged turns and wrong for a 20-question evaluation loop.
- The Copilot judge is an optional extra, never a course dependency: the mandatory path
  runs on Ollama alone.
