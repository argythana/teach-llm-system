# Guide 00f (optional): PostgreSQL + pgvector with Docker

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
On Windows it needs WSL 2; the installer walks you through it. Then check:

```bash
docker --version
docker compose version
```

## 2. Start the database

From the repository folder:

```bash
docker compose up -d       # first run downloads the image (~400 MB)
docker compose ps          # wait until STATUS shows "healthy"
```

This starts PostgreSQL 17 with the `pgvector` extension on port `5433`, user `course`,
password `course`, database `course`. These defaults are for local teaching only.

## 3. Install the Python side

```bash
uv sync --group pgvector
```

## 4. Stop it

```bash
docker compose down        # stop; the data volume stays
docker compose down -v     # stop and delete the data
```
