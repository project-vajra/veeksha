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

# Setup environment (create conda environment if needed)
echo "=== Setting up environment ==="
make setup/environment

# Activate environment
echo "=== Activating environment ==="
if [ -d "./env" ]; then
    conda activate ./env
else
    echo "Environment not found, creating it first..."
    make setup/environment
    conda activate ./env
fi

# Install dependencies
echo "=== Installing dependencies ==="
make setup/dependencies

# Build the project
echo "=== Building project ==="
make build

# Run tests
echo "=== Running tests ==="
make test

echo "=== CI setup complete ==="