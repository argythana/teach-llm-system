"""RAG building blocks, promoted from lec_02c and lec_02d.

Functions only, in the order a RAG system is built: split -> index -> retrieve -> chain.
"""

import re

_THINK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def split_documents(docs, chunk_size=800, chunk_overlap=100):
    """Split Documents into overlapping chunks (characters), keeping their metadata."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i
    return chunks


def build_chroma_index(
    chunks,
    collection_name,
    persist_dir,
    embed_model="nomic-embed-text",
    ollama_host="http://localhost:11434",
):
    """Return a persistent Chroma vector store holding ``chunks``.

    Idempotent: if the collection already holds documents, nothing is re-embedded.
    Delete ``persist_dir`` to rebuild from scratch.
    """
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_host)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    existing = store._collection.count()
    if existing == 0:
        print(f"Embedding {len(chunks)} chunks into '{collection_name}' (one-time)...")
        store.add_documents(chunks)
    else:
        print(f"Reusing '{collection_name}' with {existing} chunks from {persist_dir}")
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


def strip_think(text):
    """Remove a leaked ``<think>...</think>`` block from a reasoning model's answer."""
    return _THINK.sub("", text).strip()
