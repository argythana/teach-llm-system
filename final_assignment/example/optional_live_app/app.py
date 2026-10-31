"""Ask questions about recent ACL papers on evaluating LLM agents and LLM judges.

The app only answers: the index was built offline (build_index.py) and loaded into
PostgreSQL when the Docker image was built. Ollama and PostgreSQL run in the same
container (start.sh).
"""

import csv
import os
import time
from pathlib import Path

import gradio as gr
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector

HERE = Path(__file__).resolve().parent
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen3:1.7b")
SQL_URL = os.environ.get(
    "PAPERS_SQL_URL", "postgresql+psycopg://user@/papers?host=/tmp"
)
NUM_THREAD = int(os.environ.get("NUM_THREAD", "2"))  # a free Space has 2 vCPUs
K = 4

PROMPT = (
    "You answer questions about research papers on evaluating LLM agents and LLM "
    "judges, using only the excerpts below. If the excerpts do not contain the answer, "
    "say so. Name the paper each fact comes from.\n\n"
    "Excerpts:\n{context}\n\nQuestion: {question}"
)

store = PGVector(
    embeddings=OllamaEmbeddings(model="nomic-embed-text", num_thread=NUM_THREAD),
    collection_name="papers",
    connection=SQL_URL,
    use_jsonb=True,
)
llm = ChatOllama(
    model=CHAT_MODEL,
    reasoning=False,
    temperature=0,
    num_predict=300,  # caps the time a visitor waits, and the work per question
    num_ctx=8192,
    num_thread=NUM_THREAD,  # Ollama counts the host's CPUs, not the Space's share
)

with open(HERE / "papers.csv", newline="") as f:
    PAPERS = list(csv.DictReader(f))


def answer(question):
    """Stream the answer, then show the sources and what the answer cost."""
    started = time.perf_counter()
    docs = store.similarity_search(question, k=K)
    retrieved = time.perf_counter()

    context = "\n\n".join(
        f"[{d.metadata['title']}, page {d.metadata['page']}]\n{d.page_content}"
        for d in docs
    )
    sources = "\n".join(
        f"- [{d.metadata['title']}]({d.metadata['url']}), page {d.metadata['page']}"
        for d in docs
    )

    text, usage, first_token = "", None, None
    for chunk in llm.stream(PROMPT.format(context=context, question=question)):
        first_token = first_token or time.perf_counter()
        text += chunk.content
        usage = chunk.usage_metadata or usage
        yield text, sources, ""

    done = time.perf_counter()
    tokens_in = usage["input_tokens"] if usage else "?"
    tokens_out = usage["output_tokens"] if usage else "?"
    stats = (
        f"search {retrieved - started:.1f} s, first word after "
        f"{(first_token or done) - started:.1f} s, total {done - started:.1f} s; "
        f"tokens in {tokens_in}, out {tokens_out}; model {CHAT_MODEL}"
    )
    yield text, sources, stats


with gr.Blocks(title="Agent evaluation papers") as demo:
    gr.Markdown(
        "# Ask the papers\n"
        f"Questions about {len(PAPERS)} ACL 2025 papers on evaluating LLM agents and "
        "LLM judges, answered by a small local model on a free CPU: expect to wait."
    )
    with gr.Tab("Ask"):
        question = gr.Textbox(
            label="Question",
            placeholder="How do papers check an LLM judge against human labels?",
        )
        ask = gr.Button("Ask", variant="primary")
        answer_box = gr.Markdown()
        gr.Markdown("**Sources**")
        sources_box = gr.Markdown()
        stats_box = gr.Markdown()
        ask.click(answer, question, [answer_box, sources_box, stats_box])
        question.submit(answer, question, [answer_box, sources_box, stats_box])
    with gr.Tab("About"):
        gr.Markdown(
            "All papers are from the [ACL Anthology](https://aclanthology.org/), "
            "licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).\n\n"
            + "\n".join(f"- [{p['title']}]({p['url']}), {p['authors']}" for p in PAPERS)
        )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=7860
    )
