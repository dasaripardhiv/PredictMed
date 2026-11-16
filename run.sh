#!/usr/bin/env bash
# One-command local start (Mac/Linux). Windows: run the same steps by hand (see README).
set -e
cd "$(dirname "$0")/backend"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
[ -f .env ] || cp ../.env.example .env
uvicorn app.main:app --reload --port 8000
