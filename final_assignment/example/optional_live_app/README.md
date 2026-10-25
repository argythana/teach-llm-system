---
title: Agent Evaluation Papers
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Ask the papers (prototype)

A RAG assistant over 23 ACL Anthology 2025 papers (CC BY 4.0) on evaluating LLM agents
and LLM judges: PostgreSQL + pgvector, Ollama with `qwen3:1.7b` and `nomic-embed-text`,
and a Gradio app, in one Docker container. The block above is the header a Hugging Face
Space reads; a Docker Space needs a PRO account.

## Build the index

The image loads a dump of the index, `index.sql.gz`, which is not in git (5 MB). Build
it from the repository folder, with the course's pgvector container running
(`docker compose up -d`, guide `02a_docker_pgvector`):

1. Download the PDFs listed in `papers.csv` into `data/final_assignment_example/pdfs/`,
   named `<id>.pdf` (the ACL Anthology serves each at `https://aclanthology.org/<id>.pdf`).

1. Create the database and build the index:

   ```bash
   docker exec llm-course-pgvector createdb -U course papers
   uv run --with pypdf==6.19.0 python final_assignment/example/optional_live_app/build_index.py data/final_assignment_example/pdfs
   ```

1. Dump the two tables of the index:

   ```bash
   docker exec llm-course-pgvector pg_dump -U course -d papers --no-owner --no-privileges -t langchain_pg_collection -t langchain_pg_embedding | gzip -9 > final_assignment/example/optional_live_app/index.sql.gz
   ```

## Run it

```bash
cd final_assignment/example/optional_live_app
docker build -t paper-proto .
docker run --rm -p 7860:7860 paper-proto
```

Then open http://localhost:7860. To publish it, push this folder, `index.sql.gz`
included, to a Docker Space.
