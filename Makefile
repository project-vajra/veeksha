.PHONY: help lint format
.DEFAULT_GOAL := help

lint/black: ## check style with black
	black --check veeksha

lint/isort: ## check style with isort
	isort --check-only --profile black veeksha

lint/autoflake: ## check for unused imports
	autoflake --recursive --remove-all-unused-imports --check --exclude 'veeksha/_version.py' veeksha

lint/pyright: ## run type checking
	pyright

lint/codespell:
	codespell --skip './env/**,./docs/_build/**' -L inout

lint: lint/isort lint/black lint/autoflake lint/codespell lint/pyright	## check style

format/black: ## format code with black
	black --extend-exclude 'veeksha/lm_eval' veeksha

format/isort: ## format code with isort
	isort --profile black --extend-skip veeksha/lm_eval veeksha

format/autoflake: ## remove unused imports
	autoflake --in-place --recursive --remove-all-unused-imports --exclude 'veeksha/lm_eval/*,veeksha/_version.py' veeksha

format: format/isort format/autoflake format/black ## format code

# Test targets
test: test/unit test/functional ## Run all tests

test/unit: ## Run unit tests
	@echo "Running unit tests..."
	python -m pytest -s tests -v -m "unit" --tb=short

test/functional: ## Run functional tests
	@echo "Running functional tests..."
	python -m pytest -s tests/functional -v -m "functional and not gpu" --tb=short

test/gpu: ## Run GPU tests
	@echo "Running GPU tests..."
	python -m pytest -s tests/functional -v -m "gpu" --tb=short

test/all: ## Run all tests including GPU
	@echo "Running all tests..."
	python -m pytest -s tests -v --tb=short

# Test with coverage
test/coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	python -m pytest -s tests --cov=veeksha --cov-report=xml --cov-report=html --cov-report=term

# Rerun failed tests
test/failed-only: ## Rerun only failed tests
	@echo "Rerunning failed tests..."
	python -m pytest -s tests --lf -v --tb=short
