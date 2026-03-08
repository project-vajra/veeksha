# Veeksha

[![Publish Release to PyPI](https://github.com/project-vajra/veeksha/actions/workflows/publish_release.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/publish_release.yml) [![Deploy Documentation](https://github.com/project-vajra/veeksha/actions/workflows/deploy_docs.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/deploy_docs.yml) [![Test Suite](https://github.com/project-vajra/veeksha/actions/workflows/test_veeksha.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/test_veeksha.yml) [![Run Linters](https://github.com/project-vajra/veeksha/actions/workflows/lint.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/lint.yml)

**Veeksha** is a high-fidelity benchmarking framework for LLM inference systems.
Whether you're optimizing a production deployment, comparing serving backends, or
running capacity planning experiments, Veeksha lets you measure what matters to you:
realistic multi-turn conversations, agentic workflows, high-frequency stress tests, or targeted
microbenchmarks. One tool, any workload.

**From isolated requests to complex agentic sessions, Veeksha captures the full complexity of modern LLM workloads.**

👉 **[Why Veeksha?](https://project-vajra.github.io/veeksha/why_veeksha.html)** — Learn what sets Veeksha apart  
📚 **[Documentation](https://project-vajra.github.io/veeksha)** — Full guides and API reference

## Quick start

No install needed — run directly with [uvx](https://docs.astral.sh/uv/):

```bash
uvx veeksha benchmark \
    --client.type openai_chat_completions \
    --client.api_base http://localhost:8000/v1 \
    --client.model meta-llama/Llama-3.2-1B-Instruct \
    --traffic_scheduler.type rate \
    --traffic_scheduler.interval_generator.type poisson \
    --traffic_scheduler.interval_generator.arrival_rate 5.0 \
    --runtime.benchmark_timeout 60
```

Or use a YAML configuration file:

```bash
uvx veeksha benchmark --config my_benchmark.veeksha.yml
```

Or install with `uv pip install veeksha` / `pip install veeksha` and use `veeksha` directly.

## Installation from source

```bash
git clone https://github.com/project-vajra/veeksha.git
cd veeksha

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment (Python 3.14t recommended for true parallelism)
uv venv --python 3.14t
source .venv/bin/activate
uv pip install -e .
```
