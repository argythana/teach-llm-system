# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Purpose

A **teaching repository**, not a library or a service: 3 lectures x 2 sessions x 75
minutes on building a local LLM system with RAG and evaluation, for graduates of
[uoa_py_course](https://github.com/argythana/uoa_py_course). They know Python, pandas,
scikit-learn, Jupyter and basic MLflow; they do not know classes, type hints, pydantic,
HTTP clients, Docker, SQL, `uv`, or anything about LLMs. Fewer than half will become
programmers; all must understand what it takes to build an LLM system. The design
decisions are recorded in the planning notes; keep them unless the instructor changes
them.

## Teaching philosophy

- **Lead with the problem, then the API.** What goes wrong without this feature? What
  does it replace? Name the ML-systems concern (reproducibility, observability, cost,
  privacy).
- **Business requirements before model size.** Model choice starts from task, languages,
  license, context length, latency, hardware and privacy; hardware fit (llmfit,
  quantization) comes second. Lecture 1b is the reference; the final assignment requires
  a decision record.
- **Introduce before use.** A concept that appears in a code cell must be defined
  earlier in the same lecture or a previous one. Forward-reference later material in one
  sentence; never pre-teach it with code.
- **Define jargon once**, in a sentence or a small table, where it first appears.
- **One concept per code cell**; explicit variable names; a single imports cell at the
  top; comments explain a non-obvious choice, not what the code says.
- **Contrast pitfalls as WRONG then RIGHT cells** (`# WRONG: ...` / `# RIGHT: ...`).
- **Short cells.** At most about 150 words per Markdown cell. Bullets over prose for
  three or more items; bold or backticked eye-anchors at the start of a bullet.
- **Short notebooks.** A mandatory notebook is at most 25 code cells and 60 cells,
  walkable in 30-40 minutes. If it grows past that, it is two topics: split it.
- **Callouts are the only emojis**: `⏱ **Skip if running long.**` before a section that
  nothing downstream depends on, and `⏸ Pause point` where the class should stop and
  discuss. Two or three per mandatory notebook.
- **Idempotent cells.** Run All twice must give the same result: guard index building
  with a count check, use `exist_ok=True`, never `pip install` in a cell.
- **Write for a public reader on unknown hardware.** No "your 8 GB GPU"; timings are "in
  one example run" numbers. Offer the cpu/gpu tier choice, never bake in this machine.
- **Add an example only when it teaches something new.** A second example must contrast.
- **Link facts on first mention** (model cards on `ollama.com`/`huggingface.co`, library
  docs) and audit the links before shipping.

## Notebook skeleton

Every teaching notebook, in this order:

1. `# Lecture NNx: Title` then `**Status: Mandatory reading.**` (or
   `**Status: Optional / career-track.**`), one paragraph connecting back to the
   previous notebook, `**What this notebook covers**` bullets.
1. The imports cell, then **the configuration cell, identical in every notebook**: the
   student edits only `TIER`.
1. A "Rebuild inputs" cell when the notebook depends on earlier work, calling
   `llm_course` helpers so a student who missed a session can continue.
1. Numbered sections `## 1.` ...: Markdown motivates, code demonstrates, Markdown reads
   the output back.
1. `## Recap` bullets and a pointer to the next notebook.

Naming and layout follow uoa_py_course:
`lecture_NN_topic/reading_material/lec_NNa_topic.ipynb`, `goals_NN.md`,
`practice_exercises/lec_NN_exercises.ipynb` plus `_solutions`. Letters encode dependency
order; `a`-`d` are mandatory (two per session), `e`-`f` optional. Do not renumber.

## Environment

- Python **>= 3.12** (`.python-version` = 3.12; students have it from the Python course;
  `dspy` caps below 3.15). This diverges from teach-mlflow's 3.14 on purpose.
- **`uv` only. Never pip.** `uv sync` to install, `uv add` / `uv remove` to change
  dependencies, `uv run <tool>`; commit `uv.lock` with `pyproject.toml`. Optional
  groups: `pgvector`, `dspy`, `transformers-run` (torch from the CPU index).
- `llm_course/` is a tiny functions-only package installed by `uv sync`; every helper
  was first written live in a notebook and then promoted.

## Runtime prerequisites (system installs, not pip packages)

- **Ollama** at `http://localhost:11434` with `qwen3:1.7b` and `nomic-embed-text`
  (`qwen3:8b` for the gpu tier). Thinking is off per call (`think=False`,
  `reasoning=False`) except where deciding is the point (lec_03c).
- **MLflow tracking server** started from `mlflow_server/`:
  `uv run mlflow server --host 127.0.0.1 --port 5010`. Experiments are named
  `llm-course-01-basics`, `llm-course-02-rag`, `llm-course-03-eval`. Notebooks call
  `mlflow.tracing.disable_notebook_display()` so trace iframes do not end up in
  committed outputs. The LLM judge uses MLflow's native `ollama:/` provider, which only
  talks to the default local port.
- Optional: Docker for pgvector (`docker-compose.yml`), a Hugging Face read token in
  `.env`.
- Instructor only: the MLflow Assistant runs on the GitHub Copilot CLI through
  `tools/mlflow_server.py` (registers `llm_course/mlflow_copilot_provider.py`) after
  `tools/install_copilot_provider.py` wrote `~/.mlflow/assistant/config.json`. Never use
  the instructor's Azure OpenAI deployment for this project.

## Corpus

`corpus/uoa_py_course/` is generated by `uv run python tools/export_corpus.py --clone`
and checked by `--check` in CI. Never hand-edit it. `corpus/eval/qa_eval_set.jsonl` is
hand-written. Runtime state (Chroma index, embedding caches, Wikipedia pages) lives in
`data/`, which is gitignored.

## AI collaboration log

`ai_collaboration_log/` records the sessions in which this repository was built with
Claude Code. After each session run
`uv run python tools/export_claude_sessions.py --copy-raw` (curated Markdown and JSON
per session; raw JSONL stays gitignored) and, to mirror it into MLflow and score it,
`uv run python tools/log_session_to_mlflow.py --session <turns.json> --evaluate` plus
the blind-judge procedure in `ai_collaboration_log/evaluation/README.md`. Reflections
are the instructor's own text; draft them, do not finalise them.

## Verification

- `uv run python tools/run_all.py [--lecture N] [--optional] [--inplace]` executes the
  notebooks in order with their own folder as cwd (preflight checks Ollama and MLflow).
  Use `--inplace` to refresh committed outputs before a commit.
- `uv run prek run --all-files` runs the hooks (gitleaks, ruff on notebooks, mdformat,
  markdown wrap/justify). CI (`.github/workflows/lint.yml`) lints and import-smokes on
  Linux and Windows; it does not execute notebooks.
- Commit messages: Conventional Commits with a lecture scope, e.g. `feat(lec01): ...`.
