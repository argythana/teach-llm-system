#!/usr/bin/env python
"""Report prose and terminology problems in the course notebooks.

The rules are in CLAUDE.md ("Teaching philosophy"); ``lec_01a`` is the reference
notebook. This is a report, not a gate: some checks are heuristics, so read each hit.

Checks, one id per rule:
  para-long            a prose paragraph over 3 sentences or 60 words (one idea per paragraph)
  def-form             a bullet that starts with a bold term but has neither "is/are"
                       (a definition) nor a colon or a bold label ending in . : ? !
                       (a list bullet)
  def-multi-term       a definition bullet with more than one bold span
  def-pronoun          a definition that points back ("it", "them", "this"...) before
                       it is complete, so it cannot be read alone
  def-inline           a term defined in a prose paragraph ("**X**, a ...") instead of a bullet
  term-table           a "| Term | Meaning |" table instead of definition bullets
  term-undefined       a bold term in a recap or "What this notebook covers" list that no
                       notebook defines
  bold-emphasis        bold that is not a defined term, a list anchor, a label or a callout
  lazy-continuation    a list item's continuation line that is not indented
  blank-lines          two blank lines in a row
  single-bullet        a one-item list that is neither a definition nor a command to remember
  list-merge           definition bullets and other bullets in one rendered list: two lists
                       separated only by a blank line render as one; put a sentence between
  def-duplicate        a term defined in two places
  def-in-exercises     a definition bullet in an exercises notebook (they refer back)
  used-before-defined  a term used in an earlier notebook than the one that defines it, or a
                       mandatory notebook relying on a definition in an optional one
  def-order            a definition that uses a term defined after it (prerequisite order)
  backref-mismatch     "term (`lec_NNx`)" pointing to a notebook that does not define the term
                       while another one does
  cell-words           a Markdown cell over 160 words (the rule is about 150)
  cell-count           a mandatory notebook over 25 code cells or 60 cells

Editorial, with ``--terms`` listing every definition and the defined terms it uses: "define,
don't describe", the complete chain (every technical word a definition relies on is defined
before it) and one word, one meaning.

Usage:
    uv run python tools/notebook_prose.py [--report] [--only ID[,ID]] [NOTEBOOK ...]
    uv run python tools/notebook_prose.py --terms
    uv run python tools/notebook_prose.py --inventory
NOTEBOOK filters the report by path or short name (``lec_01a``); every notebook is still
read, because definitions and first uses cross notebooks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PARA_MAX_SENTENCES = 3
PARA_MAX_WORDS = 60
CELL_MAX_WORDS = 160  # CLAUDE.md: about 150 words per Markdown cell
MANDATORY_MAX_CELLS, MANDATORY_MAX_CODE_CELLS = 60, 25
EXERCISES_RANK = 10  # exercises come after the optional e and f notebooks

_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
_HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_WRAPPER = re.compile(r"^\s*<div[^>]*>\s*|\s*</div>\s*$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE_SPAN = re.compile(r"`+[^`]*`+")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_ARTICLE = r"(?:(?:A|An|The)\s+)?"
# - [A/An/The] **term** [(alias)] is/are ...
_DEFINITION = re.compile(
    rf"^{_ARTICLE}\*\*(?P<term>[^*]+?)\*\*(?:\s*\([^)]*\))?\s+(?:is|are)\b"
)
_ANCHOR = re.compile(rf"^\[?{_ARTICLE}\*\*")  # a linked anchor starts with "[**"
_COMMAND_ANCHOR = re.compile(r"^(\*\*)?`")
# what follows a bold term that is being defined in place
_DEFINES_AFTER = re.compile(
    r"^\s*(?:\([^)]*\)\s*)?(?:[,:—]\s+(?:a|an|the|its|one|any|which)\b|\s(?:is|are|means)\b)",
    re.IGNORECASE,
)
_POINTS_BACK = re.compile(
    r"\b(it|them|they|these|those|above|previous|this(?!\s+(?:course|notebook|lecture)))\b",
    re.IGNORECASE,
)
_ABBREV = re.compile(r"\b(e\.g|i\.e|etc|vs|cf|approx)\.", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.?!](?=\s|$)")


@dataclass
class Item:
    """One list item: its first paragraph (``text``) and where it sits."""

    indent: int
    text: str
    lazy: bool
    group: int  # items with the same group number form one list level


@dataclass
class Block:
    kind: str  # para, list, heading, table, code, quote, html
    text: str = ""
    items: list[Item] = field(default_factory=list)


@dataclass
class Cell:
    index: int
    kind: str  # markdown or code
    source: str
    blocks: list[Block] = field(default_factory=list)
    section: str = "intro"
    recap: bool = False


@dataclass
class Notebook:
    path: Path
    lecture: int
    rank: int  # 0-5 for a-f, EXERCISES_RANK for exercises
    cells: list[Cell]

    @property
    def mandatory(self) -> bool:
        return self.rank < 4

    @property
    def is_exercises(self) -> bool:
        return self.rank == EXERCISES_RANK

    @property
    def short(self) -> str:
        m = re.match(r"(lec_\d\d[a-f])_", self.path.name)
        return m.group(1) if m else self.path.stem

    @property
    def order(self) -> tuple[int, int]:
        return (self.lecture, self.rank)


@dataclass
class Definition:
    term: str
    text: str
    notebook: Notebook
    cell: int
    section: str

    @property
    def aliases(self) -> list[str]:
        names = [normalize(self.term)]
        names += [normalize(a) for a in re.findall(r"\(([^)]+)\)", self.term)]
        return [n for n in names if n]

    @property
    def body(self) -> str:
        """The definition after its bold term."""
        return _BOLD.split(self.text, maxsplit=1)[-1]


@dataclass
class Hit:
    notebook: Notebook
    cell: int
    check: str
    excerpt: str


# ---------------------------------------------------------------- parsing


def normalize(term: str) -> str:
    """Lower-case a term, drop backticks, a parenthesised alias and a plural ending."""
    t = re.sub(r"\([^)]*\)", "", term).replace("`", "").lower()
    t = " ".join(t.split())
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    return t


def _leading(line: str) -> int:
    return len(line) - len(line.lstrip())


def _starts_block(line: str) -> bool:
    s = line.strip()
    return (
        s == ""
        or s.startswith(("#", ">", "|", "<"))
        or bool(_FENCE.match(line))
        or bool(_HR.match(line))
    )


def _parse_list(lines: list[str], i: int) -> tuple[list[Item], int]:
    items: list[Item] = []
    levels: dict[int, int] = {}  # indent -> current group number
    group_count = 0
    n = len(lines)
    while i < n:
        m = _LIST.match(lines[i])
        if not m:
            break
        indent = len(m.group(1))
        content_indent = indent + len(m.group(2)) + 1
        for deeper in [k for k in levels if k > indent]:
            del levels[deeper]
        if indent not in levels:
            group_count += 1
            levels[indent] = group_count
        body, lazy, j = [m.group(3)], False, i + 1
        while j < n and not _starts_block(lines[j]) and not _LIST.match(lines[j]):
            lazy = lazy or _leading(lines[j]) < content_indent
            body.append(lines[j].strip())
            j += 1
        items.append(Item(indent, " ".join(body), lazy, levels[indent]))
        k = j
        while k < n and lines[k].strip() == "":
            k += 1
        # a list goes on after blank lines while the next line is an item, or an
        # indented paragraph that belongs to the item above
        while k < n and k > j and _leading(lines[k]) >= 2 and not _LIST.match(lines[k]):
            while k < n and lines[k].strip() != "":
                k += 1
            while k < n and lines[k].strip() == "":
                k += 1
        if k < n and _LIST.match(lines[k]):
            i = k
            continue
        return items, j
    return items, i


def parse_blocks(source: str) -> list[Block]:
    lines = source.split("\n")
    blocks: list[Block] = []
    i, n = 0, len(lines)
    while i < n:
        line, s = lines[i], lines[i].strip()
        if s == "":
            i += 1
        elif _FENCE.match(line):
            j = i + 1
            while j < n and not _FENCE.match(lines[j]):
                j += 1
            blocks.append(Block("code", "\n".join(lines[i : j + 1])))
            i = j + 1
        elif s.startswith("#"):
            blocks.append(Block("heading", s))
            i += 1
        elif s.startswith(("|", ">")):
            j = i
            while j < n and lines[j].strip().startswith(s[0]):
                j += 1
            kind = "table" if s[0] == "|" else "quote"
            blocks.append(Block(kind, "\n".join(lines[i:j])))
            i = j
        elif s.startswith("<"):
            blocks.append(Block("html", s))
            i += 1
        elif _LIST.match(line):
            items, i = _parse_list(lines, i)
            blocks.append(Block("list", items=items))
        else:
            j, para = i + 1, [s]
            while j < n and not _starts_block(lines[j]) and not _LIST.match(lines[j]):
                para.append(lines[j].strip())
                j += 1
            blocks.append(Block("para", " ".join(para)))
            i = j
    return blocks


def markdown_of(cell: dict) -> str:
    return _WRAPPER.sub("", "".join(cell["source"])).strip("\n")


def load_notebook(path: Path, lecture: int, rank: int) -> Notebook:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cells: list[Cell] = []
    section = "intro"
    for index, c in enumerate(raw["cells"]):
        if c["cell_type"] != "markdown":
            cells.append(Cell(index, "code", "".join(c["source"])))
            continue
        source = markdown_of(c)
        cell = Cell(index, "markdown", source, parse_blocks(source))
        for block in cell.blocks:
            if block.kind == "heading" and block.text.startswith("## "):
                m = re.match(r"##\s+(\d+)\.", block.text)
                section = f"§{m.group(1)}" if m else block.text[3:].strip()
        cell.section = section
        cell.recap = any(
            b.kind == "heading" and b.text.lower().startswith("## recap")
            for b in cell.blocks
        )
        cells.append(cell)
    return Notebook(path, lecture, rank, cells)


def course_notebooks() -> list[Notebook]:
    """Every course notebook, in teaching order: lecture by lecture, a to f, exercises."""
    notebooks = []
    for folder in sorted(REPO.glob("lecture_[0-9][0-9]_*")):
        lecture = int(folder.name[8:10])
        for path in sorted((folder / "reading_material").glob("lec_*.ipynb")):
            m = re.match(r"lec_\d\d([a-f])_", path.name)
            if m:
                notebooks.append(
                    load_notebook(path, lecture, "abcdef".index(m.group(1)))
                )
        for path in sorted((folder / "practice_exercises").glob("lec_*.ipynb")):
            notebooks.append(load_notebook(path, lecture, EXERCISES_RANK))
    return sorted(notebooks, key=lambda nb: (nb.order, nb.path.name))


# ---------------------------------------------------------------- text helpers


def _plain(text: str) -> str:
    return _LINK.sub(r"\1", _CODE_SPAN.sub("CODE", text))


def sentence_count(text: str) -> int:
    return len(_SENTENCE_END.findall(_ABBREV.sub(r"\1", _plain(text))))


class _Span:
    """A bold match found on masked text, read back from the original text."""

    def __init__(self, text: str, m: re.Match) -> None:
        self._text, self._start, self._end = text, m.start(), m.end()

    def start(self) -> int:
        return self._start

    def end(self) -> int:
        return self._end

    def group(self, i: int = 0) -> str:
        whole = self._text[self._start : self._end]
        return whole if i == 0 else whole[2:-2]


def bold_spans(text: str) -> list[_Span]:
    """Bold spans outside inline code (``**`` inside backticks is code, not bold)."""
    masked = _CODE_SPAN.sub(lambda m: "`" + "x" * (len(m.group()) - 2) + "`", text)
    return [_Span(text, m) for m in _BOLD.finditer(masked)]


def _excerpt(text: str, width: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _covers_lists(cell: Cell) -> set[int]:
    """Indexes of the list blocks that follow a "What this notebook covers" label."""
    found, after_label = set(), False
    for i, block in enumerate(cell.blocks):
        if block.kind == "para":
            after_label = "what this notebook covers" in block.text.lower()
        elif block.kind == "list" and after_label:
            found.add(i)
            after_label = False
    return found


def _is_definition(item: Item) -> bool:
    return bool(_DEFINITION.match(item.text))


def _definition_clause(item: Item) -> str:
    """The definition up to its first full stop, colon or semicolon after is/are."""
    rest = item.text[_DEFINITION.match(item.text).end() :]
    return re.split(r"[.:;](?=\s|$)", rest, maxsplit=1)[0]


def definitions(notebooks: list[Notebook]) -> list[Definition]:
    found = []
    for nb in notebooks:
        for cell in nb.cells:
            if cell.kind != "markdown" or cell.recap:
                continue
            covers = _covers_lists(cell)
            for i, block in enumerate(cell.blocks):
                if block.kind != "list" or i in covers:
                    continue
                for item in block.items:
                    m = _DEFINITION.match(item.text)
                    if m:
                        found.append(
                            Definition(
                                m.group("term"), item.text, nb, cell.index, cell.section
                            )
                        )
    return found


def mentions(text: str, alias: str) -> bool:
    pattern = rf"(?<![\w-]){re.escape(alias)}(?:s|es)?(?![\w-])"
    return re.search(pattern, text.replace("`", ""), re.IGNORECASE) is not None


# ---------------------------------------------------------------- checks


def check_cell(nb: Notebook, cell: Cell, defined: set[str]) -> list[Hit]:
    hits: list[Hit] = []

    def hit(check: str, text: str) -> None:
        hits.append(Hit(nb, cell.index, check, _excerpt(text)))

    words = len(re.sub(r"<[^>]+>", " ", cell.source).split())
    if words > CELL_MAX_WORDS:
        hit("cell-words", f"{words} words: {cell.source}")
    if re.search(r"\n[ \t]*\n[ \t]*\n", cell.source):
        hit("blank-lines", cell.source)

    covers = _covers_lists(cell)
    for bi, block in enumerate(cell.blocks):
        if block.kind == "table":
            header = block.text.split("\n")[0].strip("| ").split("|")[0].strip().lower()
            if header in {"term", "word"}:
                hit("term-table", block.text)
        elif block.kind == "para":
            hits += _check_paragraph(nb, cell, block.text)
        elif block.kind == "list":
            review = cell.recap or bi in covers
            top = min(item.indent for item in block.items)
            top_items = [it for it in block.items if it.indent == top]
            kinds = {_is_definition(it) for it in top_items}
            if len(kinds) == 2 and not review:
                hit("list-merge", " / ".join(it.text[:40] for it in top_items))
            if len(top_items) == 1 and not review:
                only = top_items[0]
                if not _is_definition(only) and not _COMMAND_ANCHOR.match(only.text):
                    hit("single-bullet", only.text)
            for item in block.items:
                hits += _check_item(nb, cell, item, review, defined)
    return hits


def _check_paragraph(nb: Notebook, cell: Cell, text: str) -> list[Hit]:
    hits = []
    sentences, words = sentence_count(text), len(text.split())
    if sentences > PARA_MAX_SENTENCES or words > PARA_MAX_WORDS:
        hits.append(
            Hit(
                nb,
                cell.index,
                "para-long",
                _excerpt(f"{sentences} sentences, {words} words: {text}"),
            )
        )
    callout = text.startswith(("⏱", "⏸"))
    for k, m in enumerate(bold_spans(text)):
        label = m.start() == 0 or (k == 0 and (callout or text.startswith("[**")))
        after = text[m.end() :]
        if _DEFINES_AFTER.match(after):
            hits.append(Hit(nb, cell.index, "def-inline", _excerpt(text[m.start() :])))
        elif not label:
            hits.append(
                Hit(nb, cell.index, "bold-emphasis", _excerpt(text[m.start() :]))
            )
    return hits


def _check_item(
    nb: Notebook, cell: Cell, item: Item, review: bool, defined: set[str]
) -> list[Hit]:
    hits = []

    def hit(check: str, text: str) -> None:
        hits.append(Hit(nb, cell.index, check, _excerpt(text)))

    if item.lazy:
        hit("lazy-continuation", item.text)
    bolds = bold_spans(item.text)
    anchored = bool(_ANCHOR.match(item.text))
    if _is_definition(item) and not review:
        if nb.is_exercises:
            hit("def-in-exercises", item.text)
        if len(bolds) > 1:
            hit("def-multi-term", item.text)
        if _POINTS_BACK.search(_plain(_definition_clause(item))):
            hit("def-pronoun", item.text)
        return hits
    if anchored and not review:
        after = re.sub(r"^\]\([^)]*\)", "", item.text[bolds[0].end() :])
        label = bolds[0].group(1).rstrip().endswith((".", ":", "?", "!"))
        if not label and not re.match(r"^\s*(?:\([^)]*\)\s*)?:", after):
            hit("def-form", item.text)
    for k, m in enumerate(bolds):
        if k == 0 and anchored:
            continue
        if review:
            if not any(a in defined for a in _known(m.group(1))):
                hit("term-undefined", item.text[m.start() :])
        elif _DEFINES_AFTER.match(item.text[m.end() :]):
            hit("def-inline", item.text[m.start() :])
        else:
            hit("bold-emphasis", item.text[m.start() :])
    return hits


def _known(bold_text: str) -> list[str]:
    """Normalised words and phrases of a bold span that a definition could match."""
    t = normalize(bold_text)
    words = t.split()
    phrases = {t}
    for size in range(1, len(words) + 1):
        for start in range(len(words) - size + 1):
            phrases.add(normalize(" ".join(words[start : start + size])))
    return sorted(phrases)


def cross_notebook_hits(notebooks: list[Notebook], defs: list[Definition]) -> list[Hit]:
    hits: list[Hit] = []
    first: dict[str, Definition] = {}
    for d in defs:
        key = normalize(d.term)
        if key in first:
            other = first[key]
            hits.append(
                Hit(
                    d.notebook,
                    d.cell,
                    "def-duplicate",
                    _excerpt(
                        f"also defined in {other.notebook.short} cell {other.cell}: {d.text}"
                    ),
                )
            )
        else:
            first[key] = d
    for d in first.values():
        for nb in notebooks:
            earlier = nb.order < d.notebook.order
            relies = nb.mandatory and not d.notebook.mandatory
            if nb is d.notebook or not (earlier or relies):
                continue
            for cell in nb.cells:
                if cell.kind == "markdown" and any(
                    unpointed_mention(cell.source, a, d.notebook.short)
                    for a in d.aliases
                ):
                    where = "an optional notebook" if relies else "a later notebook"
                    hits.append(
                        Hit(
                            nb,
                            cell.index,
                            "used-before-defined",
                            f"'{d.term}' is defined in {where}, {d.notebook.short} cell {d.cell}",
                        )
                    )
                    break
    for nb in notebooks:
        code = sum(c.kind == "code" for c in nb.cells)
        if nb.mandatory and (
            len(nb.cells) > MANDATORY_MAX_CELLS or code > MANDATORY_MAX_CODE_CELLS
        ):
            hits.append(
                Hit(nb, 0, "cell-count", f"{len(nb.cells)} cells, {code} code cells")
            )
    return hits


def unpointed_mention(text: str, alias: str, owner: str) -> bool:
    """A mention of ``alias`` that is not a forward gloss pointing to its owner.

    "a chat template, the recipe ... (`lec_01c` looks inside)" names the notebook that
    defines the term within 150 characters: an allowed forward mention.
    """
    flat = " ".join(text.replace("`", "").split())
    pattern = rf"(?<![\w-]){re.escape(alias)}(?:s|es)?(?![\w-])"
    return any(
        owner not in flat[max(0, m.start() - 150) : m.end() + 150]
        for m in re.finditer(pattern, flat, re.IGNORECASE)
    )


def uses(d: Definition, defs: list[Definition]) -> list[Definition]:
    """The other definitions whose term the definition ``d`` mentions."""
    own = set(d.aliases)
    return [
        e
        for e in defs
        if e is not d
        and not own & set(e.aliases)
        and any(mentions(d.body, a) for a in e.aliases)
    ]


def order_hits(defs: list[Definition]) -> list[Hit]:
    """Prerequisite order: a definition may only use terms defined before it."""
    position = {id(d): k for k, d in enumerate(defs)}
    hits = []
    for d in defs:
        for e in uses(d, defs):
            if position[id(e)] > position[id(d)]:
                hits.append(
                    Hit(
                        d.notebook,
                        d.cell,
                        "def-order",
                        f"'{d.term}' uses '{e.term}', defined later in "
                        f"{e.notebook.short} cell {e.cell}",
                    )
                )
    return hits


_BACKREF = re.compile(r"([^()]{1,80}?)\s*\(`(lec_\d\d[a-f])`\)")


def backref_hits(notebooks: list[Notebook], defs: list[Definition]) -> list[Hit]:
    """A "term (`lec_NNx`)" pointer must lead to the notebook that defines the term."""
    owner = {}
    for d in defs:
        for a in d.aliases:
            owner.setdefault(a, d)
    hits = []
    for nb in notebooks:
        for cell in nb.cells:
            if cell.kind != "markdown":
                continue
            for m in _BACKREF.finditer(" ".join(cell.source.replace("*", "").split())):
                words = normalize(re.sub(r"`", "", m.group(1))).split()[-5:]
                for size in range(len(words), 0, -1):
                    d = owner.get(normalize(" ".join(words[-size:])))
                    if d:
                        if d.notebook.short != m.group(2):
                            hits.append(
                                Hit(
                                    nb,
                                    cell.index,
                                    "backref-mismatch",
                                    f"'{d.term}' points to {m.group(2)} but is defined in "
                                    f"{d.notebook.short} cell {d.cell}",
                                )
                            )
                        break
    return hits


def all_hits(notebooks: list[Notebook]) -> list[Hit]:
    defs = definitions(notebooks)
    defined = {a for d in defs for a in d.aliases}
    hits = []
    for nb in notebooks:
        for cell in nb.cells:
            if cell.kind == "markdown":
                hits += check_cell(nb, cell, defined)
    return (
        hits
        + cross_notebook_hits(notebooks, defs)
        + order_hits(defs)
        + backref_hits(notebooks, defs)
    )


# ---------------------------------------------------------------- output


def _selected(nb: Notebook, filters: list[str]) -> bool:
    return not filters or any(
        f in (nb.short, nb.path.stem, nb.path.name) or Path(f).resolve() == nb.path
        for f in filters
    )


def print_report(notebooks: list[Notebook], filters: list[str], only: set[str]) -> None:
    hits = [
        h
        for h in all_hits(notebooks)
        if _selected(h.notebook, filters) and (not only or h.check in only)
    ]
    hits.sort(key=lambda h: (h.notebook.order, h.notebook.path.name, h.cell, h.check))
    for h in hits:
        print(f"{h.notebook.short:27} cell {h.cell:3}  {h.check:19} {h.excerpt}")
    counts: dict[str, int] = {}
    for h in hits:
        counts[h.check] = counts.get(h.check, 0) + 1
    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no hits"
    print(f"\n{len(hits)} hits: {summary}")


def print_terms(notebooks: list[Notebook]) -> None:
    """Every definition, then the defined terms it relies on (for the complete-chain read)."""
    defs = definitions(notebooks)
    for d in defs:
        where = f"{d.notebook.short} {d.section} cell {d.cell}"
        print(f"{where:30} {d.text}")
        used = ", ".join(f"{e.term} ({e.notebook.short})" for e in uses(d, defs))
        print(f"{'':30} uses: {used or 'no other defined term'}")


def print_inventory(notebooks: list[Notebook]) -> None:
    """Every candidate term with its first use and later uses (Markdown table)."""
    candidates: dict[str, dict] = {}

    def add(term: str, nb: Notebook, cell: Cell, kind: str) -> None:
        key = normalize(term)
        if not key or len(key) < 2:
            return
        entry = candidates.setdefault(
            key, {"term": term, "nb": nb, "cell": cell.index, "kinds": set()}
        )
        entry["kinds"].add(kind)

    for nb in notebooks:
        for cell in nb.cells:
            if cell.kind != "markdown" or cell.recap:
                continue
            covers = _covers_lists(cell)
            for bi, block in enumerate(cell.blocks):
                if block.kind == "para":
                    for m in bold_spans(block.text):
                        if m.start() > 0:
                            add(m.group(1), nb, cell, "prose")
                elif block.kind == "table":
                    rows = block.text.split("\n")[2:]
                    header = block.text.split("\n")[0].lower()
                    if "term" in header.split("|")[1]:
                        for row in rows:
                            term = row.strip("| ").split("|")[0]
                            add(term.replace("**", "").strip(), nb, cell, "table")
                elif block.kind == "list" and bi not in covers:
                    for item in block.items:
                        bolds = bold_spans(item.text)
                        for k, m in enumerate(bolds):
                            anchored = k == 0 and _ANCHOR.match(item.text)
                            if anchored and _is_definition(item):
                                add(m.group(1), nb, cell, "bullet")
                            elif anchored:
                                continue  # a list anchor, not a term
                            else:
                                add(m.group(1), nb, cell, "in-bullet")

    goals = {
        int(p.parts[-3][8:10]): p.read_text(encoding="utf-8")
        for p in REPO.glob("lecture_*/reading_material/goals_*.md")
    }
    print("| term | first use | how | later notebooks | goals | proposal |")
    print("| --- | --- | --- | --- | --- | --- |")
    for key, e in candidates.items():
        nb = e["nb"]
        later = [
            other.short
            for other in notebooks
            if other.order > nb.order
            and any(
                c.kind == "markdown" and mentions(c.source, key) for c in other.cells
            )
        ]
        in_goals = any(mentions(text, key) for text in goals.values())
        if "bullet" in e["kinds"]:
            proposal = "keep bullet"
        elif later or in_goals:
            proposal = f"bullet in {nb.short}"
        else:
            proposal = "inline or italics"
        shown = ", ".join(later[:6]) + (" …" if len(later) > 6 else "")
        print(
            f"| {e['term']} | {nb.short} cell {e['cell']} | {'/'.join(sorted(e['kinds']))} "
            f"| {len(later)}: {shown} | {'yes' if in_goals else ''} | {proposal} |"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--report", action="store_true", help="list the hits (default)")
    mode.add_argument("--terms", action="store_true", help="list every definition")
    mode.add_argument("--inventory", action="store_true", help="candidate terms table")
    ap.add_argument("--only", default="", help="comma-separated check ids")
    ap.add_argument("notebooks", nargs="*", help="limit the report to these notebooks")
    args = ap.parse_args()
    notebooks = course_notebooks()
    if args.terms:
        print_terms(notebooks)
    elif args.inventory:
        print_inventory(notebooks)
    else:
        only = {c.strip() for c in args.only.split(",") if c.strip()}
        print_report(notebooks, args.notebooks, only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
