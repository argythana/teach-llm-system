# Lecture 02: Embeddings, retrieval and a vector store

Two sessions. Session 3: `lec_02a` and `lec_02b`. Session 4: `lec_02c` and `lec_02d`.

## Learning Goals

### Required

- Explain what an embedding is, compute the cosine similarity of two texts, and read a
  similarity heatmap. *(lec_02a §1-3)* <!-- G1 -->
- Explain why embeddings beat keyword matching, and name their blind spots: word order,
  languages the model was not trained on, and exact names such as `GridSearchCV`.
  *(lec_02a §4-5, lec_02b §6)* <!-- G2 -->
- Build a retrieval-augmented generation (RAG) pipeline by hand: load and split
  documents, embed them once with a cache, retrieve the top `k`, and answer from them
  with a source. *(lec_02b §1-4)* <!-- G3 -->
- Trace a pipeline with MLflow: wrap its steps with `mlflow.trace` and span types, read
  each span's inputs and outputs in the UI and from code, and name the failing step of a
  wrong answer. *(lec_02b §5)* <!-- G4 -->
- Explain why documents are chunked, choose a chunk size and overlap, and split Markdown
  with a separator list that respects its structure. *(lec_02c §1-4, §6)* <!-- G5 -->
- Keep metadata with every chunk (`Document`), use it in a metadata filter, and add a
  content filter for exact terms. *(lec_02c §5, lec_02d §4)* <!-- G6 -->
- Build a persistent Chroma index idempotently and the LangChain RAG chain on top of it;
  read its autologged trace, including the token usage. *(lec_02d §1-3, §5)* <!-- G7 -->
- Measure retrieval with hit rate@k and MRR on an evaluation set with expected sources,
  and choose `k` and filters from the numbers. *(lec_02d §6)* <!-- G8 -->

### Optional / Career track

- Run the same RAG on PostgreSQL + pgvector started with Docker Compose, and query the
  vectors with SQL. *(lec_02e)* <!-- O1 -->
- Ingest an external corpus under its licence, fetch it politely, and estimate embedding
  time and storage at scale. *(lec_02f)* <!-- O2 -->

## Files

### Required

- `lec_02a_embeddings_cosine_similarity.ipynb`: embeddings, cosine similarity, a
  heatmap, meaning versus words, blind spots.
- `lec_02b_hand_built_rag_traced.ipynb`: a hand-built RAG over lectures 10-13,
  decorators, MLflow tracing, a failure diagnosed from its trace.
- `lec_02c_chunking_size_overlap_metadata.ipynb`: the embedding model's tokens, why
  chunk, a recursive splitter, overlap, metadata, chunk size.
- `lec_02d_chroma_retriever_langchain_rag.ipynb`: Chroma, idempotent ingestion, metadata
  and content filters, the LangChain RAG chain with autolog, hit rate and MRR.

### Optional / Further reading

- `lec_02e_pgvector_docker_compose.ipynb`: the same RAG on PostgreSQL + pgvector.
  Career-track value: the vector store of many production systems, next to the data a
  company already has.
- `lec_02f_wikipedia_corpus_at_scale.ipynb`: an external corpus. Career-track value:
  licences, polite fetching, and the cost of embedding at scale.
- Guides in `infra_tools/` (`02a_mlflow_server`, `02b_docker_pgvector`): the MLflow
  tracking server, and PostgreSQL + pgvector with Docker.

## Practice

`practice_exercises/lec_02_exercises.ipynb` (solutions in
`lec_02_exercises_solutions.ipynb`). Exercises are tagged with the goal they practise,
e.g. `[G5]`; `[O1]` marks stretch exercises.
