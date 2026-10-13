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
- **Definition bullets for the course's terms.** `lec_01a` is the reference notebook.
  - **Form:** `- [A/An/The] **term** is/are <kind of thing> that <what sets it apart>.`
    The "is" marks a definition; list bullets use `- **Item**: ...`.
  - **One term per bullet**, with one bold span.
  - **Self-contained:** the bullet reads alone, with no "them" or "this" pointing back.
  - **Define, don't describe:** say what the thing is; why it matters goes in the next
    sentence.
  - **Which terms:** a term gets a bullet when it comes back later (a later section or
    notebook, the goals, the recap or the exercises) or is everyday industry vocabulary.
    A one-off word gets a short inline explanation, not bold.
  - **One form only:** no inline "**X**, a ..." definitions and no `| Term | Meaning |`
    tables.
  - **Define once:** in the first notebook that uses the term, in teaching order, before
    any code uses it. A forward mention is a plain-text gloss ("a word piece, section
    3"). A mandatory notebook never relies on a definition in an optional one.
    `glossary.md` is generated from these bullets.
- **Bold means vocabulary.** Bold only for a term in its definition bullet, the anchor
  at the start of a list bullet, the Status line and the callouts; emphasis uses
  *italics*.
- **One concept per code cell**; explicit variable names; a single imports cell at the
  top; comments explain a non-obvious choice, not what the code says.
- **Contrast pitfalls as WRONG then RIGHT cells** (`# WRONG: ...` / `# RIGHT: ...`).
- **Short cells.** At most about 150 words per Markdown cell.
  - **One idea per paragraph:** a blank line wherever the paragraph's job changes (the
    main point, how it works, the lesson, an instruction, a pointer). A cause and its
    effect stay together.
  - **Paragraph limit:** at most 3 sentences and about 60 words. A long sentence becomes
    two sentences; never a trailing-space line break (the wrap hook strips it).
  - **Bullets:** over prose for three or more items, with a bold or backticked
    eye-anchor at the start. A single bullet only for a defined term or a command to
    remember (`ollama ps`).
  - **Source matches render:** indent a list item's continuation lines, and leave one
    blank line between blocks (the wrap hook does both).
- **Short notebooks.** A mandatory notebook is at most 25 code cells and 60 cells,
  walkable in 30-40 minutes. If it grows past that, it is two topics: split it.
- **Callouts are the only emojis**: `⏱ **Skip if running long.**` before a section that
  nothing downstream depends on, and `⏸ Pause point` where the class should stop and
  discuss. Two or three per mandatory notebook.
- **Idempotent cells.** Run All twice must give the same result: guard index building
  with a count check, use `exist_ok=True`, never `pip install` in a cell.
- **Write for a public reader on unknown hardware.** No "your 8 GB GPU"; timings are "in
  one example run" numbers. Offer the cpu/gpu tier choice, never bake in this machine.
- **Run-independent wording (IMPORTANT: this has broken the notebooks several times).**
  Model answers, scores, rankings, retrieved chunks, token counts, timings and ports
  change between runs, machines, and Ollama or model versions. Markdown, code comments
  and printed strings must stay true on any run: describe what to look for in the output
  ("compare what each answer says `axis` does"), not what one run printed; quote a value
  only as "in one example run"; when a behaviour may or may not happen (a question that
  retrieval sometimes misses), say so and, where it matters, show a contrasting example
  that does not depend on luck. Double-check every cell that reads an output back after
  each `--inplace` re-run.
- **Add an example only when it teaches something new.** A second example must contrast.
- **Link facts on first mention** (model cards on `ollama.com`/`huggingface.co`, library
  docs) and audit the links before shipping.

## Notebook skeleton

Every teaching notebook, in this order:

1. `# Lecture NNx: Title` then `**Status: Mandatory reading.**` (or
   `**Status: Optional / career-track.**`), one paragraph connecting back to the
   previous notebook, `**What this notebook covers**` bullets.
1. The imports cell, then **the configuration cell, identical in every notebook**: the
   student edits only `TIER`. It loads `.env` (`load_dotenv(REPO / ".env")`) before
   reading any setting and offers commented-out per-notebook overrides for `OLLAMA_HOST`
   and `MLFLOW_URI`, never for a token. Settings live in `.env`; setting environment
   variables in the terminal is explained in guide `01d_env_hugging_face` but never the
   course's pattern.
1. A "Rebuild inputs" cell when the notebook depends on earlier work, calling
   `course_helpers` helpers so a student who missed a session can continue.
1. Numbered sections `## 1.` ...: Markdown motivates, code demonstrates, Markdown reads
   the output back.
1. `## Recap` bullets and a pointer to the next notebook.

Naming and layout follow uoa_py_course:
`lecture_NN_topic/reading_material/lec_NNa_topic.ipynb`, `goals_NN.md`,
`practice_exercises/lec_NN_exercises.ipynb` plus `_solutions`. Letters encode dependency
order; `a`-`d` are mandatory (two per session), `e`-`f` optional. Do not renumber. Each
lecture also has `infra_tools/` (below), this course's own addition.

## Infrastructure guides

Each lecture has an `infra_tools/` folder with the guides for the tools and services it
needs, named `NNx_topic.md`
(`lecture_01_ollama_models_prompts_langchain/infra_tools/01c_ollama.md`); letters give
the order of use. `troubleshooting.md` at the repository root covers all lectures. Refer
to a guide by its name (`01c_ollama`), never by a bare number, which would clash with
the notebook `lec_01c`. Optional guides carry `**Status: Optional / career-track.**`
under the title. The guides are run together in class, not as homework (the README maps
each to the session that needs it). Each starts with `## Quick start`: the commands by
OS, step by step, no explanation, the slow downloads marked "leave it running". Then
`## Overview` and the detailed sections, which explain every quick-start command and are
study material. A quick start never uses a command that the detailed sections do not
explain.

The pictures in `infra_tools/screenshots/` are drawings, not captures: HTML pages in
`tools/guide_illustrations/`, rendered and compressed by
`uv run python tools/draw_guide_illustrations.py` (needs Firefox). The PNGs are
committed; rerun the script only after editing a page. Image descriptions carry no
backticks, because mdformat deletes backticked words there.
`uv run python tools/build_guide_pdfs.py` builds a PDF handout next to every guide
(needs pandoc and xelatex); the PDFs are gitignored, never committed.

## Environment

- Python **>= 3.12** (`.python-version` = 3.12; students have it from the Python course;
  `dspy` caps below 3.15). This diverges from teach-mlflow's 3.14 on purpose.
- **`uv` only. Never pip.** `uv sync` to install, `uv add` / `uv remove` to change
  dependencies, `uv run <tool>`; commit `uv.lock` with `pyproject.toml`. Optional group:
  `career-track` (pgvector packages, dspy, torch from the CPU index), installed with
  `uv sync --group career-track`.
- `course_helpers/` is a tiny functions-only package installed by `uv sync`; every
  helper was first written live in a notebook and then promoted.

## Runtime prerequisites (system installs, not pip packages)

- **Ollama** at `http://localhost:11434` with `qwen3:1.7b` and `nomic-embed-text`
  (`qwen3:8b` for the gpu tier). Thinking is off per call (`think=False`,
  `reasoning=False`) except where deciding is the point (lec_03d).
- **MLflow tracking server** started from `mlflow_server/`:
  `uv run mlflow server --host 127.0.0.1 --port 5010`. MLflow enters in lecture 3
  (`lec_03a` introduces tracing); its experiment is `llm-course-03-eval`. The
  configuration cells of lectures 1 and 2 still carry `MLFLOW_URI` and `EXPERIMENT`,
  unused, because the cell is identical everywhere. Notebooks call
  `mlflow.tracing.disable_notebook_display()` so trace iframes do not end up in
  committed outputs. The LLM judge uses MLflow's native `ollama:/` provider: the
  built-in judges (`Correctness`, `Guidelines`) only talk to the default local port,
  while `make_judge` takes `base_url` and `inference_params`.
- Optional: Docker for pgvector (`docker-compose.yml`), a Hugging Face read token in
  `.env`.
- Instructor only: the MLflow Assistant runs on the GitHub Copilot CLI through
  `tools/mlflow_server.py` (registers `course_helpers/mlflow_copilot_provider.py`) after
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
  Use `--inplace` to refresh committed outputs before a commit, then re-read every
  Markdown cell, comment and printed string that describes an output against the new
  outputs (run-independent wording, above).
- `uv run prek run --all-files` runs the hooks (gitleaks, ruff on notebooks, mdformat,
  markdown wrap/justify, glossary check). CI (`.github/workflows/lint.yml`) lints,
  import-smokes and checks the glossary on Linux and Windows; it does not execute
  notebooks.
- `uv run python tools/notebook_prose.py --report` lists prose and terminology hits
  (paragraph length, definition form, inline definitions, bold emphasis, single bullets,
  terms defined twice or used before their definition); `--terms` lists every
  definition. It is a report, not a gate: read each hit.
- `uv run python tools/build_glossary.py` regenerates `glossary.md` from the definition
  bullets; `--check` fails when it is stale or a term is defined twice. Rerun it after
  editing a definition.
- Commit messages: Conventional Commits with a lecture scope, e.g. `feat(lec01): ...`.
