"""Write pages.csv, the text of every paper page by page, for the Kaggle dataset.

Reads papers.csv next to this file and the PDFs in the folder given as argument, and
extracts the text as the worked example's section 2 does: hyphenated words rejoined,
each paper cut at its reference list. Run from the repository folder:

    uv run --with pypdf==6.19.0 python final_assignment/example/kaggle_dataset/build_pages_csv.py data/final_assignment_example/pdfs pages.csv
"""

import csv
import re
import sys
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
REFERENCES = re.compile(r"^\s*References\s*$", re.MULTILINE)


def read_pages_of(pdf_path):
    """(page number, text) for each page of a PDF, up to its reference list."""
    for number, page in enumerate(PdfReader(pdf_path).pages, start=1):
        text = re.sub(r"-\n(?=[a-z])", "", page.extract_text() or "")
        references = REFERENCES.search(text)
        if references:
            text = text[: references.start()]
        if text.strip():
            yield number, text
        if references:
            return


pdf_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
with open(HERE / "papers.csv", newline="") as f:
    paper_ids = [row["id"] for row in csv.DictReader(f)]

with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "page", "text"])
    for paper_id in paper_ids:
        for number, text in read_pages_of(pdf_dir / f"{paper_id}.pdf"):
            writer.writerow([paper_id, number, text])

print(f"{out_path}: {len(paper_ids)} papers")
