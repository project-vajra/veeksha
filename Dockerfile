FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.8.23 /uv /uvx /bin/
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*
# Copy the repository
COPY . /repo

# Set working directory
WORKDIR /repo
SHELL ["/bin/bash", "-c"]

RUN chmod +x scripts/*.sh
