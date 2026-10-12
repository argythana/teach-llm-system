# 03b DSPy: install the optional career-track group

**Status: Optional / career-track.**

## Quick start

Only the career-track notebook `lec_03f` needs this.

**Windows:** open the `teach-llm-system` folder in VS Code and choose **Terminal → New
Terminal** (guide `03a_mlflow_server`). The terminal opens in that folder.

**macOS / Linux:** the same in VS Code, or any terminal in the `teach-llm-system`
folder.

Then, on every system:

```bash
uv sync --group career-track     # the first time it downloads PyTorch: leave it running
uv run python -c "import dspy; print(dspy.__version__)"
```

The second command prints a version number, such as `3.3.1`. Restart the kernel of any
open notebook.

The sections below explain every step; they are part of the study material. If a step
fails, look up the message in [troubleshooting](../../troubleshooting.md).

## Overview

[DSPy](https://dspy.ai/) is a framework that writes the prompt for you: you declare what
goes in and what comes out, and an **optimizer** tries out prompts and examples and
keeps the ones that score best on questions you provide. `lec_01d` left it out of the
course's framework choice for that very reason (the prompt is no longer in your hands);
the optional `lec_03f` tries it on the RAG system of lecture 2 and measures what it
gains. DSPy: [source on GitHub](https://github.com/stanfordnlp/dspy) (MIT license),
[documentation](https://dspy.ai/).

DSPy is not part of the course environment that `uv sync` builds, because the mandatory
notebooks do not need it. It sits in the optional **dependency group** `career-track` of
`pyproject.toml`, with the packages of `lec_01e` and `lec_02e` (lecture 1, guide
`01a_git_uv`, "Optional packages").

## Tested versions

Checked on 12 October 2026:

| Tool                            | Version          | Check yours                                              |
| ------------------------------- | ---------------- | -------------------------------------------------------- |
| DSPy                            | 3.3.1            | `uv tree --group career-track --package dspy --depth 0`  |
| PyTorch, CPU build              | 2.14.0           | `uv tree --group career-track --package torch --depth 0` |
| `langchain-postgres`, `psycopg` | 0.0.17 and 3.3.5 | guide `02a_docker_pgvector`                              |

All four are pinned in `uv.lock`, like the course's other packages, and installed only
with `--group career-track` (section 1). **Your versions may behave differently.**
Commands, messages, defaults and results change from one version to the next, so a newer
or older version may not match this guide exactly. If something differs, compare your
version with the table first, then look in [troubleshooting](../../troubleshooting.md).
DSPy changes quickly: the names and options of its optimizers, and the prompts they
write, can differ in newer versions.

## 1. The optional group of this project

`pyproject.toml` lists the course packages under `dependencies` and two groups under
`[dependency-groups]`:

| Group          | Installed by a plain `uv sync` | Needed by                                                            |
| -------------- | ------------------------------ | -------------------------------------------------------------------- |
| `dev`          | yes (the default group)        | the checks the instructor runs before a commit                       |
| `career-track` | no                             | `lec_01e` (PyTorch), `lec_02e` (PostgreSQL client), `lec_03f` (DSPy) |

`lec_03f` also needs LiteLLM, the library DSPy uses to call models; DSPy brings it
along.

One group serves all three optional notebooks, so there is one command to remember. The
price is disk space: PyTorch alone takes about 740 MB in `.venv/` (about 200 MB to
download), even if you only want DSPy. If you synced the group for `lec_01e` or
`lec_02e` and ran no plain `uv sync` since, DSPy is already there.

`uv.lock` fixes the version of every package in every group, so everyone who syncs the
group gets the same versions.

## 2. Why the group must be named every time

`uv sync` is **exact**: it makes `.venv/` hold what you ask for in this command, and
removes everything else. A command that does not name the group asks to remove it:

```bash
# WRONG: a plain sync removes the career-track group, DSPy included
uv sync

# RIGHT: installs the course packages and the group; restores the group if it was removed
uv sync --group career-track
```

To see what a command would change before it changes anything, add `--dry-run`: uv
prints the packages it would install (`+`) and uninstall (`-`), and does nothing.

```bash
uv sync --group career-track --dry-run
```

## 3. Check the install

```bash
uv run python -c "import dspy; print(dspy.__version__)"
```

`uv run` runs the command inside `.venv/` (guide `01a_git_uv`); `python -c "..."` runs
the one line of Python between the quotes. A version number means the notebook can
import DSPy. `ModuleNotFoundError: No module named 'dspy'` means the sync did not
include the group: run `uv sync --group career-track` again.

A notebook that was open during the sync still runs the old environment: restart its
kernel (VS Code: **Restart** in the notebook toolbar).

## 4. After a `git pull`, and removing it again

A plain `uv sync`, for example after a `git pull` that changed `uv.lock`, removes the
group, DSPy included. While you still need it, sync with the same option every time:

```bash
uv sync --group career-track
```

To remove the group, run a plain `uv sync`.

## 5. What DSPy writes and sends outside the project

DSPy keeps a cache of model answers in a folder `.dspy_cache` in your home folder,
created the first time you import it. `lec_03f` switches the cache off, so that every
run really asks the model; the folder may still appear, and you can delete it at any
time.

DSPy reaches Ollama through LiteLLM, at the address in your `.env` (`OLLAMA_HOST`, guide
`01d_env_hugging_face`). The prompts and the course notes in them go only to Ollama, and
the traces to your MLflow server; no API key is needed. When LiteLLM loads, it would
also download a public list of model prices from GitHub. The line
`LITELLM_LOCAL_MODEL_COST_MAP=True` in `.env.example` turns that off: LiteLLM then uses
the copy it ships with. Copy the line into your `.env` if it is missing, and restart the
kernel.
