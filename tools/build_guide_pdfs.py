#!/usr/bin/env python
"""Build a PDF of every infrastructure guide, next to its Markdown file.

    uv run python tools/build_guide_pdfs.py                  # every lecture_*/infra_tools/*.md
    uv run python tools/build_guide_pdfs.py --lecture 1      # one lecture
    uv run python tools/build_guide_pdfs.py path/to/guide.md # chosen files

The PDFs are handouts for printing or offline reading. They are generated files:
gitignored, never committed, rebuilt after a guide changes. GitHub already shows the
Markdown with its pictures.

Needs two system programs, not Python packages: pandoc, and xelatex (TeX Live or
MiKTeX) with the fvextra and xurl packages. The default fonts are DejaVu, which have
the arrows the guides use; pass --mainfont and --monofont if they are not installed.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Long commands wrap instead of running off the page; long links may break anywhere.
LATEX_HEADER = r"""
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\},fontsize=\small}
\usepackage{xurl}
\setlength{\emergencystretch}{3em}
"""

# pandoc sizes table columns by the dashes under the Markdown header, which leaves a
# short label column too narrow for its text. Give every column a minimum share and
# take the difference from the wide ones. Handles the table objects of pandoc before
# and after 2.10.
LUA_FILTER = r"""
local MINIMUM = 0.15

local function rebalance(widths)
  local missing, spare = 0, 0
  for _, width in ipairs(widths) do
    if width == 0 then return widths end -- columns sized by their content
    if width < MINIMUM then
      missing = missing + (MINIMUM - width)
    else
      spare = spare + (width - MINIMUM)
    end
  end
  if missing == 0 or spare <= missing then return widths end
  local result = {}
  for i, width in ipairs(widths) do
    if width < MINIMUM then
      result[i] = MINIMUM
    else
      result[i] = width - (width - MINIMUM) * missing / spare
    end
  end
  return result
end

function Table(tbl)
  if tbl.colspecs then
    local widths = {}
    for i, spec in ipairs(tbl.colspecs) do widths[i] = spec[2] or 0 end
    widths = rebalance(widths)
    for i, spec in ipairs(tbl.colspecs) do
      if widths[i] ~= 0 then spec[2] = widths[i] end
    end
  else
    tbl.widths = rebalance(tbl.widths)
  end
  return tbl
end
"""


def guides(lecture=None):
    pattern = f"lecture_{lecture:02d}_*" if lecture else "lecture_*"
    return sorted(REPO.glob(f"{pattern}/infra_tools/*.md"))


def build(guide, header, lua_filter, mainfont, monofont):
    """Run pandoc in the guide's folder, so its relative picture paths resolve."""
    pdf = guide.with_suffix(".pdf")
    subprocess.run(
        [
            "pandoc",
            guide.name,
            "--output",
            pdf.name,
            # Pictures stay where the text puts them, without a floating caption.
            "--from",
            "markdown-implicit_figures",
            "--pdf-engine",
            "xelatex",
            "--include-in-header",
            str(header),
            "--lua-filter",
            str(lua_filter),
            "--highlight-style",
            "tango",
            "--variable",
            "geometry:a4paper,margin=2.2cm",
            "--variable",
            "fontsize=10pt",
            "--variable",
            f"mainfont={mainfont}",
            "--variable",
            f"monofont={monofont}",
            "--variable",
            "monofontoptions=Scale=0.9",
            "--variable",
            "colorlinks=true",
            "--variable",
            "linkcolor=blue",
            "--variable",
            "urlcolor=blue",
        ],
        cwd=guide.parent,
        check=True,
    )
    return pdf


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="*", type=Path, help="guides (default: all)")
    parser.add_argument("--lecture", type=int)
    parser.add_argument("--mainfont", default="DejaVu Sans")
    parser.add_argument("--monofont", default="DejaVu Sans Mono")
    args = parser.parse_args()

    for program in ("pandoc", "xelatex"):
        if shutil.which(program) is None:
            sys.exit(f"{program} is not on PATH; see the top of this file.")

    selected = [path.resolve() for path in args.files] or guides(args.lecture)
    if not selected:
        sys.exit("No guide found.")

    with tempfile.TemporaryDirectory() as tmp:
        header = Path(tmp) / "header.tex"
        header.write_text(LATEX_HEADER, encoding="utf-8")
        lua_filter = Path(tmp) / "table_columns.lua"
        lua_filter.write_text(LUA_FILTER, encoding="utf-8")
        for guide in selected:
            pdf = build(guide, header, lua_filter, args.mainfont, args.monofont)
            print(f"{pdf.relative_to(REPO)}  {pdf.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
