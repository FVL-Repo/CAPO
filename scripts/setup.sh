#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${CAPO_PYTHON:-python3.11}" -m venv --copies .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[pi05]'
.venv/bin/python scripts/setup_transformers.py
