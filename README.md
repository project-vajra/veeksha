# Veeksha

[![Publish Release to PyPI](https://github.com/project-vajra/veeksha/actions/workflows/publish_release.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/publish_release.yml) [![Deploy Documentation](https://github.com/project-vajra/veeksha/actions/workflows/deploy_docs.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/deploy_docs.yml) [![Test Suite](https://github.com/project-vajra/veeksha/actions/workflows/test_veeksha.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/test_veeksha.yml) [![Run Linters](https://github.com/project-vajra/veeksha/actions/workflows/lint.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/lint.yml)

**Veeksha** is a high-fidelity benchmarking framework for LLM inference systems.
Whether you're optimizing a production deployment, comparing serving backends, or
running capacity planning experiments, Veeksha lets you measure what matters to you:
realistic multi-turn conversations, agentic workflows, high-frequency stress tests, or targeted
microbenchmarks. One tool, any workload.

**Veeksha benchmarks true users, not just requests.**

Most LLM benchmarking tools measure how fast your server can process requests.
But your users don't send isolated requests. They have conversations. They think
before typing. Their agents make parallel tool calls. Their sessions have structure.
Veeksha models all of this with session graphs, flexible traffic patterns, and
composable evaluation.

👉 **[Why Veeksha?](https://project-vajra.github.io/veeksha/understanding_veeksha/why_veeksha.html)** — Learn what sets Veeksha apart  
📚 **[Documentation](https://project-vajra.github.io/veeksha)** — Full guides and API reference

## Quick Start

Install from PyPI:

```bash
pip install veeksha
```

Run a benchmark against an OpenAI-compatible endpoint:

```bash
python -m veeksha.benchmark \
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
python -m veeksha.benchmark --benchmark-config-from-file my_benchmark.veeksha.yml
```

## Installation from Source

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

## Citation

If you use our work, please consider citing our paper:

```bibtex
@misc{agrawal2024Etalon,
      title={Etalon: Holistic Performance Evaluation Framework for LLM Inference Systems}, 
      author={Amey Agrawal and Anmol Agarwal and Nitin Kedia and Jayashree Mohan and Souvik Kundu and Nipun Kwatra and Ramachandran Ramjee and Alexey Tumanov},
      year={2024},
      eprint={2407.07000},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.07000}, 
}
```

## Acknowledgement

This repository was originally created as a fork from [LLMPerf](https://github.com/ray-project/llmperf) project.
