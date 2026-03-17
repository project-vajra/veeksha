#!/usr/bin/env bash
set -euo pipefail

# Ensures the shared Python 3.14t test environment exists and is synced.

# Defaults; can be overridden by env
VENV314="${VENV314:-.venv314}"
PY314="${PY314:-3.14t}"
STAMP_FILE="${VENV314}/.veeksha-test-deps.sha256"

echo "Ensuring ${VENV314} (Python ${PY314}) exists..."
if [[ ! -f "${VENV314}/bin/activate" ]]; then
  if command -v uv >/dev/null 2>&1; then
    echo "Using uv to create ${VENV314}"
    uv venv --python "${PY314}" "${VENV314}"
  elif command -v "python${PY314}" >/dev/null 2>&1; then
    echo "Using python${PY314} venv for ${VENV314}"
    "python${PY314}" -m venv "${VENV314}"
  else
    echo "ERROR: Neither 'uv' nor 'python${PY314}' found. Please install one." >&2
    exit 1
  fi
else
  echo "${VENV314} already present."
fi

if [[ -f "uv.lock" ]]; then
  deps_hash="$(sha256sum pyproject.toml uv.lock | sha256sum | awk '{print $1}')"
else
  deps_hash="$(sha256sum pyproject.toml | awk '{print $1}')"
fi

current_hash=""
if [[ -f "${STAMP_FILE}" ]]; then
  current_hash="$(<"${STAMP_FILE}")"
fi

if [[ "${current_hash}" != "${deps_hash}" ]] || \
  ! "${VENV314}/bin/python" -c "import pytest, vidhi" >/dev/null 2>&1; then
  echo "Syncing test dependencies into ${VENV314}"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${VENV314}/bin/python" -e . --group test
  else
    # shellcheck source=/dev/null
    source "${VENV314}/bin/activate"
    python -m ensurepip --upgrade >/dev/null 2>&1 || true
    python -m pip install -U pip uv
    uv pip install -e . --group test
  fi
  printf '%s\n' "${deps_hash}" > "${STAMP_FILE}"
else
  echo "${VENV314} already has current test dependencies."
fi
