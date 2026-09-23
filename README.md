# Building a local LLM system with RAG and evaluation

A short course, 3 lectures of two 75-minute sessions each, for graduates of the
[Python for Data Science course](https://github.com/argythana/uoa_py_course) at the
University of Athens. You know Python, pandas, scikit-learn, and MLflow; you have never
built anything with a language model. By the end you will have built, on your own laptop
and with no API key, a question-answering assistant over your own course notes and
measured how good it is.

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
| 1       | 2       | `lec_01c` prompts as code and structured output · `lec_01d` LangChain chains and MLflow tracing                                                                                    |
| 2       | 3       | `lec_02a` embeddings and cosine similarity · `lec_02b` a hand-built RAG, traced                                                                                                    |
| 2       | 4       | `lec_02c` chunking · `lec_02d` Chroma, retrievers, and the LangChain RAG chain                                                                                                     |
| 3       | 5       | `lec_03a` an evaluation set and deterministic scorers · `lec_03b` LLM-as-judge and comparing variants                                                                              |
| 3       | 6       | `lec_03c` tools and agents · `lec_03d` an agentic RAG loop with LangGraph; what it takes                                                                                           |

Letters `e` and `f` in each lecture are optional career-track notebooks: running a model
with `transformers`, hosted inference, pgvector with Docker, a Wikipedia-scale corpus,
judging the judge, and DSPy prompt optimization. Each lecture has `goals_NN.md` (what
you should be able to do) and `practice_exercises/` with solutions. The course ends with
a [final assignment](final_assignment/).

## Before the first session ("homework zero")

Follow the guides in `instructions_guides/`, in order:

1. [Install uv, git and the project](instructions_guides/instruct_00a_install_uv.md):
   `git clone ...`, then `uv sync`
1. [Install Ollama and pull the models](instructions_guides/instruct_00b_install_ollama.md):
   `ollama pull qwen3:1.7b` and `ollama pull nomic-embed-text`
1. [Create a Hugging Face token](instructions_guides/instruct_00c_huggingface_token.md)
1. [Check what fits your machine with llmfit](instructions_guides/instruct_00d_llmfit_pick_a_model.md):
   `uvx llmfit recommend`
1. [Start the MLflow server](instructions_guides/instruct_00e_start_mlflow_server.md)
   (from lecture 1d on)

Then `uv run jupyter lab` and open `lecture_01_.../reading_material/lec_01a_...ipynb`.

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
