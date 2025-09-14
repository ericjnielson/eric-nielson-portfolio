#!/usr/bin/env bash
set -euo pipefail

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PORT="${PORT:-8080}"         # <- default to 8080 when PORT is empty
HOST="0.0.0.0"
echo "Starting dev server on http://${HOST}:${PORT}"
exec python app.py --host "${HOST}" --port "${PORT}"
