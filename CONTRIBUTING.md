# Contributing to Veeksha

Thank you for your interest in contributing to Veeksha. Contributions of all sizes are welcome, including bug reports, feature requests, documentation improvements, test fixes, and new functionality.

## Development setup

Veeksha requires free-threaded Python 3.14 or newer.

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/project-vajra/veeksha.git
cd veeksha

# Create and activate an environment
uv venv --python 3.14t
source .venv/bin/activate

# Install the project
uv pip install -e .
```

If you prefer, see [`README.md`](./README.md) for the latest install and usage instructions.

## Common commands

```bash
# Format Python code
make format

# Run linters
make lint

# Run unit tests
python -m pytest tests/unit/ -x -q
```

User-facing documentation lives under `docs/`. If your change affects behavior, commands, or configuration, please update the relevant docs alongside the code.

## Reporting issues

Before opening a new issue, please check whether it has already been reported. When filing an issue, include enough detail for someone else to reproduce the problem or understand the request.

Helpful details include:

- The command or config you ran
- Relevant logs or error messages
- Environment details such as Python version, OS, and server backend
- A minimal reproducer when possible

## Pull requests

When opening a pull request:

1. Keep the change focused and avoid mixing unrelated work.
2. Run `make format` and `make lint` before submitting.
3. Add or update tests when they meaningfully reduce regression risk.
4. Update `docs/` for user-facing changes.
5. Explain the motivation for the change, not just the code delta.

If your pull request fixes an issue, please reference it in the description.

## Code review

All submissions, including submissions by project members, go through code review. Small, focused pull requests are much easier to review and land quickly.

If feedback is unclear or you disagree with a suggestion, ask questions. Discussion is welcome.

## Thank you

Thanks again for helping improve Veeksha.