# 02b Docker and pgvector: a PostgreSQL vector database

**Status: Optional / career-track.**

## Quick start

Only the career-track notebook `lec_02e` needs this.

Install Docker Desktop from <https://www.docker.com/products/docker-desktop/> and open
it. Then, in a terminal, from the `teach-llm-system` folder:

```bash
docker compose up -d
docker compose ps               # wait until STATUS shows "healthy"
uv sync --group pgvector
```

When you are done:

```bash
docker compose down
```

The sections below explain every step; they are part of the study material. If a step
fails, look up the message in [troubleshooting](../../troubleshooting.md).

## Overview

Only the career-track notebook `lec_02e_pgvector_docker_compose` needs this. The
mandatory path uses Chroma, which needs no server. Skip this guide unless you want to
see what a production vector database looks like.

**Docker** runs other people's software in isolated containers, without installing it on
your system. A container is like a virtual environment for whole programs.

Docker is documented at [docs.docker.com](https://docs.docker.com/); `docker compose`,
which starts a set of containers from one file, at
[docs.docker.com/compose](https://docs.docker.com/compose/)
([source on GitHub](https://github.com/docker/compose)).

**pgvector** adds vector storage and similarity search to PostgreSQL:
[source and documentation on GitHub](https://github.com/pgvector/pgvector);
[PostgreSQL documentation](https://www.postgresql.org/docs/).

## 1. Install Docker Desktop

Download from <https://www.docker.com/products/docker-desktop/> (Windows, macOS, Linux).
On Windows it needs WSL 2; the installer walks you through it. Then check, in a terminal
from any folder:

```bash
docker --version
docker compose version
```

Docker Desktop is an application like Ollama, but the `docker` commands work only while
it is open (whale icon in the system tray or menu bar). If a command fails with an error
about connecting to the Docker daemon, open Docker Desktop and wait until it says it is
running. Whether it opens by itself after a restart is a setting: **Settings → General →
Start Docker Desktop when you sign in**.

## 2. Start the database

In a terminal, from the `teach-llm-system` folder (where `docker-compose.yml` is):

```bash
docker compose up -d       # first run downloads the image (~400 MB)
docker compose ps          # wait until STATUS shows "healthy"
```

This starts PostgreSQL 17 with the `pgvector` extension on port `5433`, user `course`,
password `course`, database `course`. These defaults are for local teaching only.

The container does **not** start again by itself after a restart of your computer or of
Docker Desktop. Run `docker compose up -d` again; the data is kept.

### If port 5433 is taken

If `docker compose up -d` fails with `port is already allocated`, another database
already uses port 5433. Put another port in your `.env` file (lecture 1, guide
`01d_env_hugging_face`):

```text
PG_PORT=5434
```

Docker Compose reads the same `.env` file as the notebooks, so this one line moves the
database and tells `lec_02e` where to find it. Run `docker compose up -d` again.

## 3. Install the Python side

From the `teach-llm-system` folder:

```bash
uv sync --group pgvector
```

A later plain `uv sync` removes these packages again (lecture 1, guide `01a_git_uv`,
"Optional packages").

## 4. Stop it

```bash
docker compose down        # stop; the data volume stays
docker compose down -v     # stop and delete the data
```
