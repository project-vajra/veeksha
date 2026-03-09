# Veeksha

High-fidelity benchmarking for LLM inference systems.

## Development Commands

- **Format**: `make format` (uses black)
- **Lint**: `make lint`
- **Test**: `python -m pytest tests/unit/ -x -q`
- **Single test file**: `python -m pytest tests/unit/path/to/test.py -x -q`

## Project Structure

- `veeksha/cli/` - CLI entry points (`commands.py` dispatch, `base.py` VeekshaCommand)
- `veeksha/config/` - Configuration dataclasses (vidhi-based: `frozen_dataclass`, `field()`, `BasePolyConfig`)
- `veeksha/microbench/` - Microbenchmark runners (prefill, decode, stress)
- `veeksha/cli/config_docs_generator.py` - Sphinx docs generator for config classes
- `tests/unit/` - Unit tests
- `tests/e2e/` - End-to-end tests

## Key Conventions

- Config system uses `vidhi` library (`frozen_dataclass`, `field()`, `BasePolyConfig`, `parse_cli_sweep`)
- CLI subcommands inherit from `VeekshaCommand(BaseCommand)` with `name=` kwarg
- Formatter is `black`; always run `make format` before committing
- Requires free-threaded Python 3.14t (GIL disabled)
- Entry point: `veeksha = "veeksha.cli.commands:main"`
