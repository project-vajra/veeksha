#!/bin/bash
# CI setup script for Vajra project
# This script contains only the core functionality required for CI pipelines

set -ex

# Get the project root directory
_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
_project_root="$(dirname "$(dirname "$(dirname "$_script_dir")")")"

# Change to project root
cd "$_project_root"

# Set CI context environment variable
export VAJRA_IS_CI_CONTEXT=1

echo "=== Vajra CI Setup ==="
echo "Project root: $_project_root"
echo "CI context: $VAJRA_IS_CI_CONTEXT"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Setup environment (create virtual environment if needed)
echo "=== Setting up environment ==="
if [ ! -d "./env" ]; then
    uv venv --python 3.12 ./env
else
    echo "Environment already exists"
fi

# Activate environment
echo "=== Activating environment ==="
source ./env/bin/activate

# Install dependencies
echo "=== Installing dependencies ==="
uv pip install -e .

# Build the project
echo "=== Building project ==="
# Build is part of uv pip install -e, no separate build needed

# Run tests
echo "=== Running tests ==="
make test

echo "=== CI setup complete ==="