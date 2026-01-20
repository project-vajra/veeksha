.PHONY: help lint format test test/unit test/e2e test/setup test/setup/314 test/failed-only coverage/report
.DEFAULT_GOAL := help

VENV314 ?= .venv314
VENV312 ?= .venv312
PY314 ?= 3.14t
PY312 ?= 3.12

lint/black: ## check style with black
	black --check veeksha

lint/isort: ## check style with isort
	isort --check-only --profile black --extend-skip veeksha/_version.py veeksha

lint/autoflake: ## check for unused imports
	autoflake --recursive --remove-all-unused-imports --check --exclude 'veeksha/_version.py' veeksha

lint/pyright: ## run type checking
	pyright

lint/codespell:
	codespell --skip './env/**,./docs/_build/**,./veeksha.egg-info/**,./test_output/**,./benchmark_results/**,./build/**,./wandb/**' -L inout

lint: lint/isort lint/black lint/autoflake lint/codespell lint/pyright	## check style

format/black: ## format code with black
	black veeksha

format/isort: ## format code with isort
	isort --profile black veeksha

format/autoflake: ## remove unused imports
	autoflake --in-place --recursive --remove-all-unused-imports --exclude 'veeksha/_version.py' veeksha

format: format/isort format/autoflake format/black ## format code

# Test targets
test: test/unit test/e2e ## Run all tests

test/setup: test/setup/314 ## Create virtual environment and install deps

test/setup/314: ## Create Python $(PY314) env for unit/lint and install dev deps
	@VENV314=$(VENV314) PY314=$(PY314) bash scripts/test_setup_314.sh

# optional: keep unit generating initial data but skip reports, or regenerate at end
test/unit: ## Run unit tests
	@echo "Running unit tests..."
	@VENV314=$(VENV314) bash scripts/run_tests_unit.sh

test/e2e: ## Run end-to-end tests
	@echo "Running e2e tests..."
	@VENV314=$(VENV314) bash scripts/run_tests_e2e.sh

# Emit final coverage reports into mounted test_output directory
coverage/report:
	coverage xml -o test_output/python_coverage.xml
	coverage html -d test_output/python_coverage_html

# Rerun failed tests
test/failed-only: ## Rerun only failed tests
	@echo "Rerunning failed tests..."
	python -m pytest -s tests --lf -v --tb=short
