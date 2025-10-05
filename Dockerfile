FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

# Install wget for uv installation
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Install uv
RUN wget -qO- https://astral.sh/uv/install.sh | sh
ENV PATH="$HOME/.cargo/bin:$PATH"

# Copy the repository
COPY . /repo

# Set working directory
WORKDIR /repo

# Set CI context
ENV VAJRA_IS_CI_CONTEXT=1

# Create virtual environment
RUN uv venv --python 3.12 ./env

# Activate environment and install dependencies
RUN source ./env/bin/activate && uv pip install -e .

# The container is now ready for running tests or other commands