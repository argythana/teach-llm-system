"""RAG building blocks, promoted from lec_02c and lec_02d.

Functions only, in the order a RAG system is built: split -> index -> retrieve ->
chain -> measure.
"""

import re

_THINK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_HEADING = re.compile(r"^#{1,6} +(.+)$")

# Separators tried in order (lec_02c): headings first, then paragraphs, lines, words.
# Only "## " and "### ": in the notes' code blocks, "# " starts a Python comment.
HEADINGS_FIRST = ["\n## ", "\n### ", "\n\n", "\n", " ", ""]

# Chunk kinds that count as course notes: lecture notebooks and setup guides, but not
# exercises, solutions, goals or the "read_agents" readings (see corpus.kind_of).
LECTURE_KINDS = ["lecture", "guide"]


def _headings(text):
    """Return ``[(offset, heading)]`` for the Markdown headings of ``text``.

    Lines inside fenced code blocks are skipped: there ``#`` starts a comment.
    """
    found, offset, in_code = [], 0, False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        # A fence line opens or closes a block; ```{ }``` inline on one line does not.
        if stripped.startswith("```") and stripped.count("```") == 1:
            in_code = not in_code
        elif not in_code:
            match = _HEADING.match(line.strip())
            if match:
                found.append((offset, match.group(1).strip()))
        offset += len(line)
    return found


def split_documents(docs, chunk_size=800, chunk_overlap=100, min_chars=80):
    """Split Markdown Documents into overlapping chunks, keeping their metadata.

    The splitter cuts at headings first, then paragraphs, lines and words (lec_02c).
    Chunks shorter than ``min_chars`` (a heading or a rule left alone) are dropped: they
    answer nothing, and their generic vectors sit close to every short question. Each
    chunk gets ``start_index`` (its position in its file) and ``heading`` (the nearest
    Markdown heading above it) in its metadata.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=HEADINGS_FIRST,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = []
    for doc in docs:
        headings = _headings(doc.page_content)
        for chunk in splitter.split_documents([doc]):
            start = chunk.metadata["start_index"]
            above = [h for offset, h in headings if offset <= start]
            chunk.metadata["heading"] = above[-1] if above else ""
            if len(chunk.page_content) >= min_chars:
                chunks.append(chunk)
    return chunks


def build_chroma_index(
    chunks,
    collection_name,
    persist_dir,
    embed_model="nomic-embed-text",
    ollama_host="http://localhost:11434",
    batch_size=256,
):
    """Return a persistent Chroma vector store holding ``chunks``.

    Idempotent: a collection that already holds exactly ``len(chunks)`` chunks is reused;
    an empty, half-built or out-of-date one is rebuilt. Ids are ``source#start_index``,
    stable while the file is unchanged. After changing the embedding model, delete
    ``persist_dir``: the count cannot see that change.
    """
    from pathlib import Path

    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_host)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},  # scores = 1 - cosine
    )
    where = Path(persist_dir).name
    existing = len(store.get(include=[])["ids"])
    if existing == len(chunks):
        print(f"Reusing '{collection_name}': {existing} chunks in {where}/")
        return store
    if existing:
        print(
            f"'{collection_name}' holds {existing} chunks, not {len(chunks)}: rebuilding"
        )
        store.reset_collection()
    print(f"Embedding {len(chunks)} chunks into '{collection_name}' (one-time)...")
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        ids = [f"{c.metadata['source']}#{c.metadata['start_index']}" for c in batch]
        store.add_documents(batch, ids=ids)
        print(f"  {start + len(batch)} / {len(chunks)}")
    return store


def format_context(docs):
    """Turn retrieved Documents into the text block the prompt receives."""
    return "\n\n".join(
        f"[source: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
    )


DEFAULT_RAG_PROMPT = (
    "You answer questions about a Python course using only the notes below. "
    "If the notes do not contain the answer, say so. Mention the source file you used.\n\n"
    "Notes:\n{context}\n\nQuestion: {question}"
)


def make_rag_chain(retriever, llm, prompt_template=None):
    """Build ``question -> retrieve -> prompt -> llm -> text`` as one LangChain Runnable."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    prompt = ChatPromptTemplate.from_template(prompt_template or DEFAULT_RAG_PROMPT)
    return (
        {"context": retriever | format_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def source_rank(docs, expected_source):
    """Position (1 = first) of the first Document from ``expected_source``, or None."""
    for position, doc in enumerate(docs, start=1):
        if doc.metadata.get("source") == expected_source:
            return position
    return None


def hit_rate_and_mrr(ranks, ks=(1, 2, 4, 8)):
    """Summarise ``source_rank`` results: hit rate at each k, and the MRR."""
    n = len(ranks)
    summary = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks}
    summary["MRR"] = sum(1 / r for r in ranks if r) / n
    return summary


def strip_think(text):
    """Remove a leaked ``<think>...</think>`` block from a reasoning model's answer."""
    return _THINK.sub("", text).strip()
