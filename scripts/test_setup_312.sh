#!/usr/bin/env bash
set -euo pipefail

# sets up a Python 3.12 environment for functional/gpu tests

# Defaults; can be overridden by env
VENV312="${VENV312:-.venv312}"
PY312="${PY312:-3.12}"

echo "Ensuring ${VENV312} (Python ${PY312}) exists..."
if [[ ! -f "${VENV312}/bin/activate" ]]; then
  if command -v uv >/dev/null 2>&1; then
    echo "Using uv to create ${VENV312}"
    uv venv --python "${PY312}" "${VENV312}"
    # shellcheck source=/dev/null
    source "${VENV312}/bin/activate"
    uv pip install -e ".[test]"
  elif command -v "python${PY312}" >/dev/null 2>&1; then
    echo "Using python${PY312} venv for ${VENV312}"
    "python${PY312}" -m venv "${VENV312}"
    # shellcheck source=/dev/null
    source "${VENV312}/bin/activate"
    pip install -U pip
    pip install -e ".[test]"
  else
    echo "WARNING: Neither 'uv' nor 'python${PY312}' found. Functional/GPU tests will try current env and may skip." >&2
  fi
else
  echo "${VENV312} already present."
fi
