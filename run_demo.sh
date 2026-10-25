#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
if [ ! -f .venv/demo_installed.ok ]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  touch .venv/demo_installed.ok
fi
python -m uvicorn app:app --host 0.0.0.0 --port 8000
