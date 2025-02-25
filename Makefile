.PHONY: help lint format
.DEFAULT_GOAL := help

lint/black: ## check style with black
	black --check veeksha

lint/isort: ## check style with isort
	isort --check-only --profile black veeksha

lint/autoflake: ## check for unused imports
	autoflake --recursive --remove-all-unused-imports --check veeksha

lint/pyright: ## run type checking
	pyright

lint/codespell:
	codespell --skip './env/**,./docs/_build/**' -L inout

lint: lint/isort lint/black lint/autoflake lint/codespell lint/pyright	## check style

format/black: ## format code with black
	black veeksha

format/isort: ## format code with isort
	isort --profile black veeksha

format/autoflake: ## remove unused imports
	autoflake --in-place --recursive --remove-all-unused-imports veeksha

format: format/isort format/autoflake format/black ## format code
