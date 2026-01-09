#!/usr/bin/env bash
set -euo pipefail

VENV314="${VENV314:-.venv314}"

if [[ -f "${VENV314}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV314}/bin/activate"
else
  echo "NOTE: ${VENV314} not found, using current Python. Consider running: make test/setup"
fi

python -Xgil=0 -m pytest -s tests -v -m "unit" --tb=short \
  --junitxml=test_output/pytest-unit-results.xml \
  --cov=veeksha --cov-report=
