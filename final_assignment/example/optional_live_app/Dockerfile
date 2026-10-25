# One container, three programs: PostgreSQL + pgvector, Ollama, and the Gradio app.

# Ollama's official image, without its GPU libraries (a free Space has no GPU).
FROM ollama/ollama:0.34.0 AS ollama
RUN rm -rf /usr/lib/ollama/cuda_* /usr/lib/ollama/mlx_* /usr/lib/ollama/vulkan

FROM pgvector/pgvector:pg17

# Certificates, so that ollama pull can check the download site.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Ollama's program and libraries from the stage above; uv from its own image.
COPY --from=ollama /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/lib/ollama
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

# Spaces run the container as user 1000.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:/usr/lib/postgresql/17/bin:$PATH \
    PGDATA=/home/user/pgdata \
    PGHOST=/tmp \
    OLLAMA_MODELS=/home/user/.ollama/models
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN uv venv --python 3.12 && uv pip install --no-cache -r requirements.txt

# The models go into the image, so a restarted Space downloads nothing.
RUN ollama serve > /dev/null 2>&1 & \
    until ollama list > /dev/null 2>&1; do sleep 1; done; \
    ollama pull qwen3:1.7b && ollama pull nomic-embed-text

# The index, built offline, is loaded into a database inside the image.
COPY --chown=user index.sql.gz .
RUN initdb --auth=trust --username=user > /dev/null && \
    pg_ctl --wait --options="-k /tmp -c listen_addresses=''" start && \
    createdb papers && psql --quiet -d papers -c "CREATE EXTENSION vector" && \
    gunzip -c index.sql.gz | psql --quiet -d papers > /dev/null && \
    pg_ctl --wait stop && rm index.sql.gz

COPY --chown=user app.py papers.csv start.sh ./
EXPOSE 7860
CMD ["bash", "start.sh"]
