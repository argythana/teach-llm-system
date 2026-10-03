# Building a local LLM system with RAG and evaluation

A short course, 3 lectures of two 75-minute sessions each, for graduates of the
[Python for Data Science course](https://github.com/argythana/uoa_py_course) at the
University of Athens.

## About the course

This is a new official course of the
[Business Information Systems](https://bis-analytics.econ.uoa.gr/) postgraduate program
of the University of Athens, offered as part of its Research Seminars Series.

You know Python, pandas, scikit-learn, and MLflow; you have never built anything with a
language model. By the end you will have built, on your own laptop and with no API key,
a question-answering assistant over your own course notes and measured how good it is.

## What you will build

```text
question ──► retriever (Chroma) ──► prompt ──► local model (Ollama) ──► answer
                 ▲                                        │
        corpus pipeline: notes → chunks → embeddings      ▼
                                                 MLflow: traces, evaluation, comparison
```

The stack is the one used in industry today, in its open-source form: **Ollama** serves
the model, **LangChain** connects the parts, **Chroma** stores the embeddings,
**MLflow** records what happened and scores the answers, and **LangGraph** adds a
decision loop at the end.

## Sessions

| Lecture | Session | Notebooks (`reading_material/`)                                                                                                                                                    |
| ------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1       | 1       | `lec_01a` your laptop as an LLM server: uv, Ollama, tokens, context, temperature · `lec_01b` choosing a model: requirements first, then the Hugging Face Hub, quantization, llmfit |
| 1       | 2       | `lec_01c` prompts as code and structured output · `lec_01d` LangChain chains                                                                                                       |
| 2       | 3       | `lec_02a` embeddings and cosine similarity · `lec_02b` a hand-built RAG, traced with MLflow                                                                                        |
| 2       | 4       | `lec_02c` chunking · `lec_02d` Chroma, filters, the LangChain RAG chain, and measuring retrieval                                                                                   |
| 3       | 5       | `lec_03a` an evaluation set and deterministic scorers · `lec_03b` LLM-as-judge and comparing variants                                                                              |
| 3       | 6       | `lec_03c` tools and agents · `lec_03d` an agentic RAG loop with LangGraph; what it takes                                                                                           |

Letters `e` and `f` in each lecture are optional career-track notebooks: running a model
with `transformers`, hosted inference, pgvector with Docker, a Wikipedia-scale corpus,
judging the judge, and DSPy prompt optimization. Each lecture has `goals_NN.md` (what
you should be able to do) and `practice_exercises/` with solutions. The course ends with
a [final assignment](final_assignment/).

## Setup, together in class

Setting up the tools is part of the course, not homework: we run it together in class,
one guide at a time. Each lecture keeps its guides in an `infra_tools/` folder, next to
`reading_material/` and `practice_exercises/`. Each guide starts with a **Quick start**,
the commands for your system without explanation; the rest of the guide explains them
and is study material.

| When                          | Guide                                                                                                    | You end with                             |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| Session 1, start              | [`01a_git_uv`](lecture_01_ollama_models_prompts_langchain/infra_tools/01a_git_uv.md)                     | the course folder and its environment    |
| Session 1, start              | [`01b_llmfit`](lecture_01_ollama_models_prompts_langchain/infra_tools/01b_llmfit.md)                     | your tier: `cpu` or `gpu`                |
| Session 1, start              | [`01c_ollama`](lecture_01_ollama_models_prompts_langchain/infra_tools/01c_ollama.md)                     | Ollama running, the course models pulled |
| Session 1, before `lec_01b`   | [`01d_env_hugging_face`](lecture_01_ollama_models_prompts_langchain/infra_tools/01d_env_hugging_face.md) | your settings file with a read token     |
| Session 3, before `lec_02b`   | [`02a_mlflow_server`](lecture_02_embeddings_rag_vector_store/infra_tools/02a_mlflow_server.md)           | the tracking server running              |
| Only for `lec_02e` (optional) | [`02b_docker_pgvector`](lecture_02_embeddings_rag_vector_store/infra_tools/02b_docker_pgvector.md)       | a PostgreSQL vector database             |

`uv sync` and the model downloads take several minutes when a whole class downloads at
once, so we start them first and talk while they run. If your connection is slow, you
may run the quick starts of `01a_git_uv` to `01c_ollama` before class; nothing is lost
if you do not. When something fails, look up the message in
[troubleshooting](troubleshooting.md).

After `01c_ollama`, open the first notebook in VS Code (guide `01a_git_uv`):
`lecture_01_ollama_models_prompts_langchain/reading_material/lec_01a_uv_ollama_first_call_tokens.ipynb`.

Hardware: any laptop with 8 GB of RAM runs the default `cpu` tier (`qwen3:1.7b`). Every
notebook starts with a configuration cell where a GPU owner can switch to the `gpu`
tier. Windows, macOS and Linux are all fine; no Docker is needed for the mandatory path.

## The corpus

`corpus/uoa_py_course/` holds the Python course's own notebooks exported to Markdown. It
is the dataset of this course: you will ask questions you already know the answers to,
which is the only way to judge whether a retrieval system is telling the truth. See
`corpus/LICENSE_NOTE.md`.

## How this course was built

The course was designed and written with an AI coding assistant. `ai_collaboration_log/`
holds the curated transcripts of every session, the instructor's reflections on the
collaboration, and an automated evaluation of the assistant's replies that uses the same
MLflow tooling lecture 3 teaches.

## For instructors and contributors

`CLAUDE.md` documents the teaching principles, conventions, and how to regenerate the
corpus and re-execute the notebooks (`tools/run_all.py`).

## License

Code, including notebook code cells, is licensed under the [MIT License](LICENSE). Prose
(Markdown files and notebook text cells) is licensed under
[CC BY 4.0](LICENSE-CC-BY-4.0.txt). Parts of the prose are adapted from
[teach-mlflow](https://github.com/argythana/teach-mlflow) by Thanasis Argyriou, licensed
under CC BY 4.0, with changes.
