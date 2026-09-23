"""Helper functions for the teach-llm-system course.

Every function here was first written live in a notebook and then "promoted" into
this package so later notebooks can rebuild their inputs in one cell:

- ``llm_course.checks`` : is Ollama up, are the models pulled, is MLflow reachable?
- ``llm_course.corpus`` : load the course notes (and optional Wikipedia pages).
- ``llm_course.rag``    : split, index (Chroma), and build the RAG chain.
"""
