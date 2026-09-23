#!/usr/bin/env python
"""Export the uoa_py_course notes into corpus/uoa_py_course/ as Markdown.

    uv run python tools/export_corpus.py --src /path/to/uoa_py_course   # export from a clone
    uv run python tools/export_corpus.py --clone                        # shallow-clone it first
    uv run python tools/export_corpus.py --check                        # verify corpus matches manifest

Standard library only, so it also runs before `uv sync`. Notebook cells become
Markdown (see llm_course.corpus.notebook_to_markdown); .py and .txt files are fenced.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from llm_course.corpus import notebook_to_markdown  # noqa: E402

SOURCE_REPO = "https://github.com/argythana/uoa_py_course"
OUT_DIR = REPO / "corpus" / "uoa_py_course"
MANIFEST = REPO / "corpus" / "manifest.json"
LECTURE_GROUPS = (
    "lectures_01_06_fundamentals_for_data_science",
    "lectures_07_13_pandas_plots_scikit",
)
SUBFOLDERS = ("reading_material", "practice_exercises")
SKIP_DIRS = {
    "archive",
    "lecture_archive",
    "outdated_files",
    "simple_cases",
    "notebooks_2024",
}
EXTENSIONS = {".ipynb", ".py", ".md", ".txt"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(path):
    if path.suffix == ".ipynb":
        return notebook_to_markdown(path)
    text = path.read_text(encoding="utf-8", errors="replace").rstrip()
    if path.suffix == ".md":
        return text + "\n"
    lang = "python" if path.suffix == ".py" else "text"
    return f"# {path.name}\n\n```{lang}\n{text}\n```\n"


def source_files(src):
    for group in LECTURE_GROUPS:
        for lecture_dir in sorted((src / group).glob("lecture_[0-9][0-9]_*")):
            for sub in SUBFOLDERS:
                for path in sorted((lecture_dir / sub).rglob("*")):
                    if path.suffix not in EXTENSIONS or not path.is_file():
                        continue
                    if SKIP_DIRS & set(path.relative_to(lecture_dir).parts):
                        continue
                    yield lecture_dir, path


def export(src):
    commit = subprocess.run(
        ["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if OUT_DIR.exists():
        for old in OUT_DIR.rglob("*.md"):
            old.unlink()
    entries = []
    for lecture_dir, path in source_files(src):
        lecture = re.match(r"lecture_(\d{2})", lecture_dir.name).group(1)
        out = OUT_DIR / f"lecture_{lecture}" / f"{path.stem}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        rel_source = path.relative_to(src).as_posix()
        text = f"<!-- source: {rel_source} @ {commit[:12]} -->\n\n" + convert(path)
        out.write_text(text, encoding="utf-8")
        entries.append(
            {
                "path": out.relative_to(REPO / "corpus").as_posix(),
                "source": rel_source,
                "chars": len(text),
                "sha256": sha256(out),
            }
        )
    manifest = {
        "source_repo": SOURCE_REPO,
        "commit": commit,
        "exported_on": date.today().isoformat(),
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    total = sum(e["chars"] for e in entries)
    print(
        f"Exported {len(entries)} files ({total / 1e6:.2f} M chars) from {SOURCE_REPO} @ {commit[:12]} -> {OUT_DIR}"
    )


def check():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    listed = {e["path"]: e["sha256"] for e in manifest["files"]}
    on_disk = {
        p.relative_to(REPO / "corpus").as_posix(): sha256(p)
        for p in OUT_DIR.rglob("*.md")
    }
    problems = [f"missing: {p}" for p in listed if p not in on_disk]
    problems += [f"not in manifest: {p}" for p in on_disk if p not in listed]
    problems += [
        f"changed: {p}" for p in listed if p in on_disk and on_disk[p] != listed[p]
    ]
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(
        f"corpus OK: {len(listed)} files match manifest (commit {manifest['commit'][:12]})"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, help="path to a uoa_py_course clone")
    parser.add_argument(
        "--clone",
        action="store_true",
        help="shallow-clone uoa_py_course into a temp dir first",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify corpus/ matches corpus/manifest.json",
    )
    args = parser.parse_args()
    if args.check:
        check()
        return
    if args.clone:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "uoa_py_course"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    SOURCE_REPO,
                    str(dest),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(dest), "sparse-checkout", "set", *LECTURE_GROUPS],
                check=True,
            )
            export(dest)
        return
    if not args.src:
        parser.error("give --src PATH or --clone")
    export(args.src)


if __name__ == "__main__":
    main()
