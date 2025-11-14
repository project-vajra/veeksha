#!/usr/bin/env bash
set -euo pipefail

# sets up a Python 3.14t environment for unit/lint tests

# Defaults; can be overridden by env
VENV314="${VENV314:-.venv314}"
PY314="${PY314:-3.14t}"

echo "Ensuring ${VENV314} (Python ${PY314}) exists..."
if [[ ! -f "${VENV314}/bin/activate" ]]; then
  if command -v uv >/dev/null 2>&1; then
    echo "Using uv to create ${VENV314}"
    uv venv --python "${PY314}" "${VENV314}"
    # shellcheck source=/dev/null
    source "${VENV314}/bin/activate"
    uv pip install -e ".[dev]"
  elif command -v "python${PY314}" >/dev/null 2>&1; then
    echo "Using python${PY314} venv for ${VENV314}"
    "python${PY314}" -m venv "${VENV314}"
    # shellcheck source=/dev/null
    source "${VENV314}/bin/activate"
    pip install -U pip
    pip install -e ".[dev]"
  else
    echo "ERROR: Neither 'uv' nor 'python${PY314}' found. Please install one." >&2
    exit 1
  fi
else
  echo "${VENV314} already present."
fi
