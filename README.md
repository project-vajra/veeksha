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

In a fresh environment:

Install from PyPI:

```bash
pip install veeksha
```

Run a benchmark against an OpenAI-compatible endpoint:

```bash
python -Xgil=0 -m veeksha.benchmark \
    --client-type openai_chat_completions \
    --openai-chat-completions-client-api-base http://localhost:8000/v1 \
    --openai-chat-completions-client-model meta-llama/Llama-3.2-1B-Instruct \
    --traffic-scheduler-type rate \
    --rate-traffic-scheduler-interval-generator-type poisson \
    --rate-traffic-scheduler-poisson-interval-generator-arrival-rate 5.0 \
    --runtime-benchmark-timeout 60
```

Or use a YAML configuration file:

```bash
python -Xgil=0 -m veeksha.benchmark --benchmark-config-from-file my_benchmark.veeksha.yml
```

## Installation from source

```bash
git clone https://github.com/project-vajra/veeksha.git
cd veeksha

mamba env create -p ./env -f environment.yml
mamba activate ./env
```
