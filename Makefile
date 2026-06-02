.PHONY: help lint format test test/unit test/e2e test/setup test/setup/314 test/failed-only coverage/report setup/environment setup/update-environment setup/activate format/clang format/cpp lint/clang-format lint/cpplint lint/cpp
.DEFAULT_GOAL := help

VENV314 ?= .venv314
VENV312 ?= .venv312
PY314 ?= 3.14t
PY312 ?= 3.12
ENV_BIN ?= ./env/bin
CLANG_FORMAT ?= $(if $(wildcard $(ENV_BIN)/clang-format),$(ENV_BIN)/clang-format,clang-format)
CPPLINT ?= $(if $(wildcard $(ENV_BIN)/cpplint),$(ENV_BIN)/cpplint,cpplint)
CPP_FILES := $(shell find native/stt_probe_cpp -type f \( -name '*.cpp' -o -name '*.cc' -o -name '*.h' \) -not -path '*/build/*' 2>/dev/null)

setup/environment: ## create repo-local mamba/conda development environment
	@if command -v mamba >/dev/null 2>&1; then \
		mamba env create -f environment-dev.yml -p ./env; \
	elif command -v conda >/dev/null 2>&1; then \
		conda env create -f environment-dev.yml -p ./env; \
	else \
		echo "Neither mamba nor conda found." >&2; \
		exit 1; \
	fi

setup/update-environment: ## update repo-local mamba/conda development environment
	@if command -v mamba >/dev/null 2>&1; then \
		mamba env update -f environment-dev.yml -p ./env --prune; \
	elif command -v conda >/dev/null 2>&1; then \
		conda env update -f environment-dev.yml -p ./env --prune; \
	else \
		echo "Neither mamba nor conda found." >&2; \
		exit 1; \
	fi

setup/activate: ## show command to activate environment
	@echo "To activate the environment, run:"
	@if [ -d "./env" ]; then \
		echo "  conda activate ./env"; \
	else \
		echo "  Environment not found. Run 'make setup/environment' first."; \
	fi

lint/black: ## check style with black
	black --check veeksha

lint/isort: ## check style with isort
	isort --check-only --profile black --extend-skip veeksha/_version.py veeksha

lint/autoflake: ## check for unused imports
	autoflake --recursive --remove-all-unused-imports --check --exclude 'veeksha/_version.py' veeksha

lint/pyright: ## run type checking
	pyright

lint/clang-format: ## check native C++ format with clang-format
	@failed=0; \
	for file in $(CPP_FILES); do \
		$(CLANG_FORMAT) --dry-run --Werror "$$file" >/dev/null 2>&1 || { \
			echo "clang-format check failed for: $$file"; \
			failed=1; \
		}; \
	done; \
	exit $$failed

lint/cpplint: ## run native C++ style checks with cpplint
	$(CPPLINT) --recursive \
		--exclude=native/stt_probe_cpp/build \
		--filter="-build/include_what_you_use,-build/c++11,-whitespace/parens,-whitespace/braces,-runtime/references,-readability/namespace,-whitespace/indent,-legal/copyright" \
		native/stt_probe_cpp/src

lint/cpp: lint/clang-format lint/cpplint ## check native C++ style

lint/codespell:
	codespell --skip './env/**,./docs/_build/**,./veeksha.egg-info/**,./test_output/**,./benchmark_results/**,./build/**,./wandb/**' -L inout

lint: lint/isort lint/black lint/autoflake lint/codespell lint/pyright lint/cpp	## check style

format/black: ## format code with black
	black veeksha

format/isort: ## format code with isort
	isort --profile black veeksha

format/autoflake: ## remove unused imports
	autoflake --in-place --recursive --remove-all-unused-imports --exclude 'veeksha/_version.py' veeksha

format/clang: ## format native C++ code with clang-format
	$(CLANG_FORMAT) -i $(CPP_FILES)

format/cpp: format/clang ## format native C++ code

format: format/isort format/autoflake format/black format/cpp ## format code

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
