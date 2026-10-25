"""Build the papers index in PostgreSQL + pgvector, once, offline (prototype script).

Reads papers.csv and the PDFs, splits each paper into chunks, embeds them with
nomic-embed-text through the local Ollama and stores them in the `papers` database of
the course's pgvector container (lec_02e). The Space never runs this: it loads the dump
written by `docker exec ... pg_dump` (see README).
"""

import csv
import os
import re
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PDF_DIR = Path(sys.argv[1])
SQL_URL = os.environ.get(
    "PAPERS_SQL_URL", "postgresql+psycopg://course:course@localhost:5433/papers"
)
REFERENCES = re.compile(r"^\s*References\s*$", re.MULTILINE)


def pages_of(paper):
    """One Document per page, up to the reference list, which answers nothing."""
    reader = PdfReader(PDF_DIR / f"{paper['id']}.pdf")
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"-\n(?=[a-z])", "", text)  # join words hyphenated at line ends
        match = REFERENCES.search(text)
        if match:
            text = text[: match.start()]
        if text.strip():
            yield Document(
                page_content=text,
                metadata={
                    "source": paper["id"],
                    "title": paper["title"],
                    "url": paper["url"],
                    "page": number,
                },
            )
        if match:
            return


with open(HERE / "papers.csv", newline="") as f:
    papers = list(csv.DictReader(f))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100, add_start_index=True
)
chunks = [
    chunk
    for paper in papers
    for chunk in splitter.split_documents(list(pages_of(paper)))
    if len(chunk.page_content) >= 80
]
print(f"{len(papers)} papers, {len(chunks)} chunks")

store = PGVector(
    embeddings=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="papers",
    connection=SQL_URL,
    use_jsonb=True,
    pre_delete_collection=True,  # a rebuild starts from an empty collection
)
for start in range(0, len(chunks), 256):
    batch = chunks[start : start + 256]
    ids = [
        f"{c.metadata['source']}#p{c.metadata['page']}#{c.metadata['start_index']}"
        for c in batch
    ]
    store.add_documents(batch, ids=ids)
    print(f"  {start + len(batch)} / {len(chunks)}")
