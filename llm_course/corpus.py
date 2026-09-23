"""The course corpus: the uoa_py_course notes exported to Markdown.

``notebook_to_markdown`` is the exporter used by ``tools/export_corpus.py``
(standard library only). ``load_corpus`` turns the exported files into LangChain
``Document`` objects with metadata (lec_02b). ``fetch_wikipedia_pages`` is the
optional, larger corpus for lec_02f.
"""

import json
import re
from pathlib import Path

_DATA_IMAGE = re.compile(
    r"!\[[^\]]*\]\(data:image[^)]*\)|<img[^>]*src=\"data:[^\"]*\"[^>]*>"
)
_ATTACHMENT = re.compile(r"!\[[^\]]*\]\(attachment:[^)]*\)")
_LECTURE_DIR = re.compile(r"lecture_(\d{2})_")


def notebook_to_markdown(path):
    """Return a notebook's markdown and code cells as one Markdown string.

    Markdown cells are copied verbatim (embedded base64 images are replaced by
    ``[image]``); code cells become fenced ``python`` blocks; outputs are dropped.
    """
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    parts = []
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


def kind_of(filename):
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


def load_corpus(corpus_dir="corpus/uoa_py_course", lectures=None):
    """Load the exported course notes as LangChain Documents.

    ``lectures`` limits the load, e.g. ``[10, 11, 12, 13]``. Each Document carries
    ``metadata = {"source", "lecture", "kind", "title"}``.
    """
    from langchain_core.documents import Document

    corpus_dir = Path(corpus_dir)
    docs = []
    for path in sorted(corpus_dir.rglob("*.md")):
        match = _LECTURE_DIR.search(str(path))
        lecture = int(match.group(1)) if match else 0
        if lectures is not None and lecture not in lectures:
            continue
        text = path.read_text(encoding="utf-8")
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


def fetch_wikipedia_pages(titles, out_dir="data/wikipedia", lang="en"):
    """Download Wikipedia articles as plain text and return them as Documents.

    Uses the MediaWiki API directly (no extra library). Text is CC BY-SA 4.0;
    each file starts with an attribution header. Cached: a page already on disk
    is not fetched again.
    """
    import requests
    from langchain_core.documents import Document

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": "teach-llm-system/0.1 (course material; contact via GitHub)"
    }
    docs = []
    for title in titles:
        path = out_dir / (re.sub(r"[^A-Za-z0-9_-]+", "_", title) + ".md")
        if not path.exists():
            response = requests.get(
                f"https://{lang}.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts|info",
                    "explaintext": 1,
                    "redirects": 1,
                    "format": "json",
                    "titles": title,
                },
                headers=headers,
                timeout=30,
            )
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
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"wikipedia/{path.name}",
                    "lecture": 0,
                    "kind": "wikipedia",
                    "title": title,
                },
            )
        )
    return docs
