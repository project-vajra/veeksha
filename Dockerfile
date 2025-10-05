FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.8.23 /uv /uvx /bin/

# Copy the repository
COPY . /repo

# Set working directory
WORKDIR /repo
SHELL ["/bin/bash", "-c"]  

# Set CI context
ENV VAJRA_IS_CI_CONTEXT=1

# Create virtual environment
RUN uv venv --python 3.12 ./env

# Activate environment and install dependencies
RUN source ./env/bin/activate && uv pip install -e .

# The container is now ready for running tests or other commands