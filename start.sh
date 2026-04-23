#!/bin/bash
cd "$(dirname "$0")"
# Load local .env if present (TAPO_USER, TAPO_PASS, etc.)
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi
./.venv/bin/python app.py
