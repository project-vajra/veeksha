#!/usr/bin/env bash
set -euo pipefail

VENV314="${VENV314:-.venv314}"

if [[ -f "${VENV314}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV314}/bin/activate"
else
  echo "NOTE: ${VENV314} not found, using current Python. Consider running: make test/setup"
fi

mkdir -p test_output

python -Xgil=0 -m pytest -s tests/e2e -v --tb=short \
  --junitxml=test_output/pytest-e2e-results.xml
