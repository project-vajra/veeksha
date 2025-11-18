#!/usr/bin/env bash
set -euo pipefail

# functional tests that require GPU
# the 3.12 env runs the vllm server
# the 3.14 env runs the veeksha tests over the vllm server

VENV314="${VENV314:-.venv314}"
VENV312="${VENV312:-.venv312}"

if [[ -f "${VENV314}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV314}/bin/activate"
else
  echo "NOTE: ${VENV314} not found, using current Python. Consider running: make test/setup"
fi

if [[ -x "${VENV312}/bin/python" ]]; then
  export VLLM_PYTHON="$(pwd)/${VENV312}/bin/python"
else
  echo "NOTE: ${VENV312} not found, VLLM_PYTHON unset; tests may skip. Consider running: make test/setup"
fi

python -Xgil=0 -m pytest -s tests/functional -v -m "gpu" --tb=short \
  --junitxml=test_output/pytest-functional-gpu-results.xml \
  --cov=veeksha --cov-append --cov-report=
