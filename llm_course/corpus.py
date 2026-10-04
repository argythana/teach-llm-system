"""The course corpus: the uoa_py_course notes exported to Markdown.

``notebook_to_markdown`` is the exporter used by ``tools/export_corpus.py``
(standard library only). ``load_sections`` is lec_02b's split at ``## `` headings;
``load_corpus`` turns the exported files into LangChain ``Document`` objects with
metadata (lec_02d); ``load_eval_set`` reads the evaluation set (lec_02d).
``fetch_wikipedia_pages`` is the optional, larger corpus for lec_02f.
"""

from __future__ import annotations

import json
import re
from collections.abc import Container, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # imported inside the functions at run time, to keep imports light
    from langchain_core.documents import Document

_DATA_IMAGE = re.compile(
    r"!\[[^\]]*\]\(data:image[^)]*\)|<img[^>]*src=\"data:[^\"]*\"[^>]*>"
)
_ATTACHMENT = re.compile(r"!\[[^\]]*\]\(attachment:[^)]*\)")
_LECTURE_DIR = re.compile(r"lecture_(\d{2})")
_WIKI_SOURCE = re.compile(r"Source: (\S+) \(revision (\d+)\)")
_EXPORT_NOTE = re.compile(r"\A<!-- source:.*?-->\s*", flags=re.DOTALL)


def notebook_to_markdown(path: str | Path) -> str:
    """Return a notebook's markdown and code cells as one Markdown string.

    Markdown cells are copied verbatim (embedded base64 images are replaced by
    ``[image]``); code cells become fenced ``python`` blocks; outputs are dropped.
    """
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", [])).rstrip()
        if not source:
            continue
        if cell.get("cell_type") == "markdown":
            source = _DATA_IMAGE.sub("[image]", source)
            source = _ATTACHMENT.sub("[image]", source)
            parts.append(source)
        elif cell.get("cell_type") == "code":
            parts.append(f"```python\n{source}\n```")
    return "\n\n".join(parts) + "\n"


def kind_of(filename: str | Path) -> str:
    """Classify a corpus file by the uoa_py_course naming convention."""
    name = Path(filename).name
    if name.startswith("goals_"):
        return "goals"
    if name.startswith("read_"):
        return "reading"
    if name.startswith("instruct_"):
        return "guide"
    if "_solutions" in name:
        return "solutions"
    if "_exercises" in name:
        return "exercises"
    return "lecture"


def load_corpus(
    corpus_dir: str | Path = "corpus/uoa_py_course",
    lectures: Container[int] | None = None,
) -> list[Document]:
    """Load the exported course notes as LangChain Documents.

    ``lectures`` limits the load, e.g. ``[10, 11, 12, 13]``. Each Document carries
    ``metadata = {"source", "lecture", "kind", "title"}``.
    """
    from langchain_core.documents import Document

    corpus_dir = Path(corpus_dir)
    docs: list[Document] = []
    for path in sorted(corpus_dir.rglob("*.md")):
        match = _LECTURE_DIR.search(path.parent.name)
        lecture = int(match.group(1)) if match else 0
        if lectures is not None and lecture not in lectures:
            continue
        # The exporter's provenance note (<!-- source: ... -->) belongs in metadata,
        # not in the text, where its words would match questions (lec_02c).
        text = _EXPORT_NOTE.sub("", path.read_text(encoding="utf-8"))
        heading = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path.relative_to(corpus_dir)),
                    "lecture": lecture,
                    "kind": kind_of(path.name),
                    "title": heading.group(1).strip() if heading else path.stem,
                },
            )
        )
    return docs


def load_sections(
    corpus_dir: str | Path = "corpus/uoa_py_course",
    lectures: Iterable[int] = (10, 11, 12, 13),
) -> list[dict[str, str]]:
    """Split the course notes at their ``## `` headings, as lec_02b does by hand.

    Returns a list of dicts ``{"source", "heading", "text"}``, one per section.
    """
    corpus_dir = Path(corpus_dir)
    sections: list[dict[str, str]] = []
    for lecture in lectures:
        for path in sorted((corpus_dir / f"lecture_{lecture:02d}").glob("*.md")):
            text = path.read_text(encoding="utf-8")
            for part in re.split(r"(?m)^(?=## )", text):
                if part.strip():
                    sections.append(
                        {
                            "source": f"lecture_{lecture:02d}/{path.name}",
                            "heading": next(
                                (ln for ln in part.splitlines() if ln.startswith("#")),
                                "",
                            )[:80],
                            "text": part,
                        }
                    )
    return sections


def load_eval_set(
    path: str | Path = "corpus/eval/qa_eval_set.jsonl",
) -> list[dict[str, Any]]:
    """Read the evaluation set (lec_02d): one JSON object per line."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def fetch_wikipedia_pages(
    titles: Iterable[str], out_dir: str | Path = "data/wikipedia", lang: str = "en"
) -> list[Document]:
    """Download Wikipedia articles as plain text and return them as Documents.

    Uses the MediaWiki API directly (no extra library). Text is CC BY-SA 4.0;
    each file starts with an attribution header. Cached: a page already on disk
    is not fetched again. Polite to the API: a User-Agent with a contact URL, one
    second between requests, and a wait-and-retry when told "429 Too Many Requests".
    """
    import time

    import requests
    from langchain_core.documents import Document

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": "teach-llm-system/0.1 (https://github.com/argythana/teach-llm-system)"
    }
    docs: list[Document] = []
    for title in titles:
        path = out_dir / (re.sub(r"[^A-Za-z0-9_-]+", "_", title) + ".md")
        if not path.exists():
            for attempt in range(4):
                time.sleep(1)  # at most one request per second
                response = requests.get(
                    f"https://{lang}.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "prop": "extracts|info",
                        "explaintext": "1",
                        "redirects": "1",
                        "format": "json",
                        "titles": title,
                    },
                    headers=headers,
                    timeout=30,
                )
                if response.status_code != 429:
                    break
                time.sleep(int(response.headers.get("Retry-After", 10)))
            response.raise_for_status()
            page = next(iter(response.json()["query"]["pages"].values()))
            if "extract" not in page:
                raise ValueError(f"No Wikipedia page found for {title!r}")
            url = f"https://{lang}.wikipedia.org/wiki/{page['title'].replace(' ', '_')}"
            header = (
                f"# {page['title']}\n\n"
                f"Source: {url} (revision {page.get('lastrevid')}). "
                "Text by Wikipedia contributors, licensed CC BY-SA 4.0.\n\n"
            )
            path.write_text(header + page["extract"] + "\n", encoding="utf-8")
        text = path.read_text(encoding="utf-8")
        origin = _WIKI_SOURCE.search(text)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"wikipedia/{path.name}",
                    "lecture": 0,
                    "kind": "wikipedia",
                    "title": title,
                    "url": origin.group(1) if origin else "",
                    "revision": origin.group(2) if origin else "",
                    "license": "CC BY-SA 4.0",
                },
            )
        )
    return docs
