#!/usr/bin/env bash
# Start the database and the model server inside the container, then the app.
set -euo pipefail
pg_ctl --wait --log=/tmp/postgres.log --options="-k /tmp -c listen_addresses=''" start
ollama serve > /tmp/ollama.log 2>&1 &
until ollama list > /dev/null 2>&1; do sleep 1; done
exec python app.py
