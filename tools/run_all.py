#!/usr/bin/env python
"""Execute the course notebooks in order, the way a student would run them.

    uv run python tools/run_all.py                    # mandatory notebooks (letters a-d), all lectures
    uv run python tools/run_all.py --lecture 2        # one lecture
    uv run python tools/run_all.py --optional         # include the career-track notebooks (e, f)
    uv run python tools/run_all.py --inplace          # refresh the committed outputs
    uv run python tools/run_all.py --tier gpu         # run with the GPU tier (COURSE_TIER env var)

Preflight checks Ollama, the models, and (from lecture 2 on) the MLflow server, then
runs each notebook with its own folder as the working directory via
`jupyter nbconvert --execute`.
Executed copies go to _executed/ (gitignored) unless --inplace.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODELS = {
    "cpu": ["qwen3:1.7b", "nomic-embed-text"],
    "gpu": ["qwen3:8b", "nomic-embed-text"],
}
MANDATORY_LETTERS = "abcd"
NOTEBOOK = re.compile(r"lec_(\d{2})([a-z])_")


def notebooks(lecture=None, optional=False):
    for path in sorted(REPO.glob("lecture_*/reading_material/lec_*.ipynb")):
        match = NOTEBOOK.match(path.name)
        if not match:
            continue
        number, letter = int(match.group(1)), match.group(2)
        if lecture and number != lecture:
            continue
        if not optional and letter not in MANDATORY_LETTERS:
            continue
        yield path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lecture", type=int)
    parser.add_argument("--optional", action="store_true")
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument(
        "--tier", choices=["cpu", "gpu"], default=os.environ.get("COURSE_TIER", "cpu")
    )
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per cell")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    os.environ["COURSE_TIER"] = args.tier
    # The same settings file the notebooks' configuration cell reads.
    load_dotenv(REPO / ".env")
    todo = list(notebooks(args.lecture, args.optional))
    if not args.skip_preflight:
        from llm_course.checks import check_mlflow, check_ollama

        check_ollama(
            os.environ.get("OLLAMA_HOST", "http://localhost:11434"), MODELS[args.tier]
        )
        # Lecture 1 does not use MLflow; tracing starts in lec_02b.
        if any(int(NOTEBOOK.match(path.name).group(1)) >= 2 for path in todo):
            check_mlflow(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5010"))

    print(f"Running {len(todo)} notebooks (tier={args.tier}, optional={args.optional})")
    failures = []
    for path in todo:
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            f"--ExecutePreprocessor.timeout={args.timeout}",
            path.name,
        ]
        if args.inplace:
            cmd.append("--inplace")
        else:
            out_dir = REPO / "_executed" / path.parents[1].name
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--output-dir", str(out_dir)]
        started = time.time()
        result = subprocess.run(cmd, cwd=path.parent)
        status = "ok" if result.returncode == 0 else "FAILED"
        print(
            f"{status:6} {path.parents[1].name}/{path.name}  ({time.time() - started:.0f}s)"
        )
        if result.returncode != 0:
            failures.append(path)
    if failures:
        print("\nFailed:\n" + "\n".join(str(p.relative_to(REPO)) for p in failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
