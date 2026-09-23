# Guide 00a: install `uv` and set up the course project

In the Python course you created a virtual environment with `python -m venv` and
installed packages with `pip install -r requirements.txt`. This course uses **`uv`**,
one tool that does both jobs, can also install Python itself, and records the exact
versions it installed in `uv.lock`, so every student and the instructor run the same
code.

To explore `uv`: [source on GitHub](https://github.com/astral-sh/uv),
[documentation](https://docs.astral.sh/uv/).

This guide goes in order: install `uv`, install `git`, download the course, see how `uv`
creates virtual environments, then let `uv sync` build the course environment.

## 1. Install uv

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen the terminal, then check:

```bash
uv --version
```

If the command is not found on Windows, the installer printed the folder it used
(usually `C:\Users\<you>\.local\bin`); add it to your PATH or reopen the terminal.

## 2. Install git

[`git`](https://git-scm.com/) is the version-control tool the course repository lives
in. With it you download the course once and later fetch the instructor's updates with
one command, instead of downloading a new ZIP each week.

- **Windows:** `winget install --id Git.Git -e`, or the installer from
  [git-scm.com](https://git-scm.com/downloads/win) (the default options are fine).
- **macOS:** `xcode-select --install`, or `brew install git` if you use Homebrew.
- **Linux (Debian/Ubuntu):** `sudo apt install git`.

Close and reopen the terminal, then check:

```bash
git --version
```

## 3. Get the course files

In the folder where you keep your course work:

```bash
git clone https://github.com/argythana/teach-llm-system.git
cd teach-llm-system
```

When the instructor pushes new material, run `git pull` inside this folder.

Without git: download the repository as a ZIP from GitHub (green **Code** button) and
unzip it. You will have to download it again for every update.

## 4. How uv creates a virtual environment

`uv` has its own command for what `python -m venv` did in the Python course:

```bash
uv venv --python 3.13
```

Two differences matter:

- **You choose the Python version.** `python -m venv` can only copy the Python you ran
  it with. `uv venv --python 3.13` uses 3.13, and if it is not on your computer, `uv`
  downloads it first. `uv python list` shows the versions available.
- **The folder is always `.venv/`** in the current directory, unless you give another
  name (`uv venv my_env`).

You can try it in an empty scratch folder and delete the folder afterwards. **You do not
need to run it for this course**: the next step does it for you.

## 5. Create the course environment with `uv sync`

The instructor has already decided the environment and pushed the decision to git, in
three files in the repository:

| File              | What it pins                                              |
| ----------------- | --------------------------------------------------------- |
| `.python-version` | the Python version (3.12)                                 |
| `pyproject.toml`  | the packages the course needs, like `requirements.txt`    |
| `uv.lock`         | the exact version of every package, dependencies included |

From inside the `teach-llm-system` folder, run:

```bash
uv sync
```

`uv sync` reads these three files and does everything in one command: it downloads
Python 3.12 if you do not have it, creates `.venv/` with that version (the `uv venv`
step of section 4), and installs exactly the versions in `uv.lock` (about 2 GB, a few
minutes the first time). Running it again only fixes what differs from the lock, so it
is safe to rerun, for example after a `git pull`.

The whole workflow, next to what you did in the Python course:

| venv + pip (Python course)            | uv (this course)                            |
| ------------------------------------- | ------------------------------------------- |
| `python -m venv course_venv`          | `uv venv --python 3.12` (done by `uv sync`) |
| `pip install -r requirements.txt`     | `uv sync` (creates `.venv/` and installs)   |
| `pip install some-package`            | `uv add some-package`                       |
| activate, then `jupyter lab`          | `uv run jupyter lab`                        |
| run a tool once without installing it | `uvx llmfit`                                |

## 6. Open the notebooks with the right interpreter

The notebooks run in [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/)
([source on GitHub](https://github.com/jupyterlab/jupyterlab)), which `uv sync`
installed.

`.venv/` holds the Python interpreter that has the course packages. Which interpreter a
notebook uses depends on how you open it:

- **JupyterLab from the terminal (recommended).** Run, from the `teach-llm-system`
  folder:

  ```bash
  uv run jupyter lab
  ```

  `uv run <command>` runs the command inside `.venv/`, so JupyterLab and its kernel
  already use the right interpreter. Use `uv run` for every tool in this course
  (`uv run jupyter lab`, `uv run mlflow server ...`).

- **VS Code.** Open the `teach-llm-system` folder, open a notebook, and click **Select
  Kernel** (top right) → **Python Environments** → the entry that points to
  `teach-llm-system/.venv`. VS Code usually proposes it first. Without this step the
  notebook runs on whatever Python VS Code found last, and the first `import` fails.

- **A terminal session.** Activating the environment as in the Python course also works:
  `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows), then
  `jupyter lab`.

To check, run this in a notebook cell; the path must contain `teach-llm-system/.venv`:

```python
import sys

print(sys.executable)
```

## Optional packages

Two career-track notebooks need extra packages that most students will not install:

```bash
uv sync --group pgvector          # lec_02e, needs Docker too
uv sync --group transformers-run  # lec_01e, downloads PyTorch (~200 MB)
```

## Never use pip in this project

`pip install` inside `.venv/` bypasses `uv.lock`, so your environment silently differs
from everyone else's. If a notebook needs a package that is missing, the fix is
`uv add <package>` (which also updates `pyproject.toml` and `uv.lock`), not pip.
