#!/usr/bin/env bash
set -euo pipefail

VENV314="${VENV314:-.venv314}"

if [[ -f "${VENV314}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV314}/bin/activate"
else
  echo "NOTE: ${VENV314} not found, using current Python."
fi

python -m pytest -s tests -v --tb=short \
  --junitxml=test_output/pytest-all-results.xml


