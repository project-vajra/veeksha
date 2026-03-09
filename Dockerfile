FROM ubuntu:24.04
COPY --from=ghcr.io/astral-sh/uv:0.8.23 /uv /uvx /bin/
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
# Copy the repository
COPY . /repo

# Set working directory
WORKDIR /repo
SHELL ["/bin/bash", "-c"]

RUN chmod +x scripts/*.sh
