"""Helper functions for the teach-llm-system course.

Every function here was first written live in a notebook and then "promoted" into
this package so later notebooks can rebuild their inputs in one cell:

- ``course_helpers.checks`` : is Ollama up, are the models pulled, is MLflow reachable?
- ``course_helpers.corpus`` : load the course notes (and optional Wikipedia pages).
- ``course_helpers.rag``    : split, index (Chroma), and build the RAG chain.
- ``course_helpers.evaluation`` : evaluation records, scorers, the LLM judge, comparisons.
- ``course_helpers.agents``     : the course's tools and the agentic RAG graph.
"""
