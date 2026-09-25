# 01d .env and Hugging Face: your settings file and a read token

## Quick start

In a browser:

1. Log in at <https://huggingface.co> and open <https://huggingface.co/settings/tokens>.
1. **Create new token**, role **Read**, name `teach-llm-system`, create it, and copy it.

In a terminal, from the `teach-llm-system` folder:

```powershell
Copy-Item .env.example .env          # Windows PowerShell
```

```bash
cp .env.example .env                 # macOS / Linux
```

Open `.env` in VS Code or another plain-text editor, paste the token after `HF_TOKEN=`,
and save.

The sections below explain every step; they are part of the study material. If a step
fails, look up the message in [troubleshooting](../../troubleshooting.md).

## Overview

The **Hugging Face Hub** is where open models live (you used it in the Python course to
deploy a Gradio app to a Space). Lecture 1b searches it through its Python API to choose
a model. Public information works without logging in, but anonymous requests are
rate-limited and some models (for example Meta's Llama family) require you to accept a
license first, which needs an account. A **read token** solves both.

The token is the first entry in **`.env`**, the one file where this course keeps your
personal settings (section 2).

The Hub is documented at
[huggingface.co/docs/hub](https://huggingface.co/docs/hub/index). The Python library
that queries it, `huggingface_hub`, is installed by `uv sync`:
[source on GitHub](https://github.com/huggingface/huggingface_hub),
[documentation](https://huggingface.co/docs/huggingface_hub/index).

## 1. Create the token

1. Log in at <https://huggingface.co> (create an account if you deleted yours).
1. Open <https://huggingface.co/settings/tokens>, click **Create new token**.
1. Choose the **Read** role, name it `teach-llm-system`, create it, and copy it once.

## 2. Your settings file `.env`

Some values differ from student to student and do not belong in the notebooks: a token
is a secret, and the address of Ollama or MLflow depends on your computer. The course
keeps them in one plain-text file, `.env`, in the `teach-llm-system` folder, one
`NAME=value` per line:

| Setting                                             | Do you need it?                                 | What it is                                                             |
| --------------------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------- |
| `HF_TOKEN`                                          | yes, from lecture 1b                            | your Hugging Face read token                                           |
| `OLLAMA_HOST`                                       | only if Ollama runs on another computer or port | where Ollama answers (guide `01c_ollama`)                              |
| `MLFLOW_TRACKING_URI`                               | only if you start MLflow on another port        | where the MLflow server answers (lecture 2, guide `02a_mlflow_server`) |
| `HOSTED_BASE_URL`, `HOSTED_API_KEY`, `HOSTED_MODEL` | optional, you add them in `lec_01f`             | a hosted model provider                                                |

Why a file:

- **Secrets stay out of the code.** `.env` is listed in `.gitignore`, so git never
  uploads it, and a notebook you share never contains your token.
- **One place for every notebook.** Change a value once, and every notebook uses it the
  next time its configuration cell runs.
- **It stays.** It survives restarts and works the same in JupyterLab and VS Code, on
  every operating system.

### Create it

The course ships a template, `.env.example`. Copy it in a terminal, from the
`teach-llm-system` folder:

```bash
cp .env.example .env                 # macOS / Linux
Copy-Item .env.example .env          # Windows PowerShell
```

Open `.env` in VS Code or any plain-text editor (not Word) and paste the token after
`HF_TOKEN=`:

```text
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

Leave the other lines as they are unless a guide tells you to change them. A name that
starts with a dot hides the file in the macOS Finder (`Cmd+Shift+.` shows it) and in
`ls` (`ls -a` shows it); VS Code's file panel always shows it.

### How the notebooks read it

The configuration cell at the top of every notebook contains:

```python
load_dotenv(REPO / ".env")  # your settings file: OLLAMA_HOST, HF_TOKEN, ...
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
```

An **environment variable** is a named value the operating system keeps for a running
program, outside its code. `load_dotenv` copies each line of `.env` into one;
`os.environ.get` reads one back, with a default for when `.env` does not set it. Some
libraries read theirs directly: `huggingface_hub` finds `HF_TOKEN` without your code
naming it. Without a `.env` file the notebooks still run on the defaults; only the cells
that need the token complain. `load_dotenv` comes from `python-dotenv`, installed by
`uv sync`: [source on GitHub](https://github.com/theskumar/python-dotenv).

`load_dotenv` does not replace a value it already loaded. If you edit `.env` while a
notebook is open, restart the kernel (**Kernel → Restart Kernel**) and run the cells
again.

## 3. Three ways to set a value, and which to use

| Way                 | Where you write it                           | Applies to                                            | When                                      |
| ------------------- | -------------------------------------------- | ----------------------------------------------------- | ----------------------------------------- |
| **1. `.env`**       | a line in `.env`                             | every notebook, until you change the line             | **always: this is the course's pattern**  |
| **2. One notebook** | a line in that notebook's configuration cell | that notebook only                                    | trying another address; **never a token** |
| **3. The terminal** | a command typed before `uv run jupyter lab`  | notebooks started from that terminal, until it closes | not recommended                           |

### 1. In `.env` (recommended)

Section 2. Use it for every setting, including the token.

### 2. In one notebook

The configuration cell ends its settings with:

```python
# To override .env in this notebook only, uncomment and edit (never a token):
# OLLAMA_HOST = "http://192.168.1.20:11434"
# MLFLOW_URI = "http://127.0.0.1:5011"
```

Delete the `# ` in front of a line, edit the address, and run the cell again. The line
runs after `.env` is read, so it wins, in this notebook only. Put the `# ` back to
return to `.env`. Never type a token in a notebook: the notebook is a file you save,
share and maybe push to GitHub, and the token would travel with it.

### 3. In the terminal (not recommended)

You will meet this in other tutorials, so here is what it looks like:

```bash
export OLLAMA_HOST=http://192.168.1.20:11434      # macOS / Linux
uv run jupyter lab
```

```powershell
$env:OLLAMA_HOST = "http://192.168.1.20:11434"    # Windows PowerShell
uv run jupyter lab
```

This course does not use it, because:

- **It is invisible.** Nothing in the project shows the value; a week later nobody
  remembers it was set.
- **It is short-lived and local.** It lasts until that terminal window closes, and it
  does not reach notebooks opened in VS Code or from another terminal.
- **The syntax differs** between macOS/Linux, PowerShell and the old Windows `cmd`.
- **It silently beats `.env`.** `load_dotenv` does not replace a variable that already
  exists, so a value left in the terminal (or in a startup file such as `~/.bashrc`)
  wins, and editing `.env` seems to do nothing.

When a setting seems ignored, print it in the notebook (`print(OLLAMA_HOST)`), then
check the terminal: `echo $OLLAMA_HOST` (macOS/Linux) or `echo $env:OLLAMA_HOST`
(PowerShell) should print an empty line.

One exception: the `ollama` command-line tool does not read `.env`, so guide
`01c_ollama` sets `OLLAMA_HOST` in the terminal for a one-off test of that command. That
is a test, not a course setting.

## 4. If a token ever leaks

Open the tokens page, click **Invalidate and refresh** (or delete it) and create a new
one. Tokens are credentials, like passwords: never paste one into a notebook cell, a
chat, or a screenshot.
