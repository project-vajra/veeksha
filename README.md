# Veeksha

[![Publish Release to PyPI](https://github.com/project-vajra/veeksha/actions/workflows/publish_release.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/publish_release.yml) [![Deploy Documentation](https://github.com/project-vajra/veeksha/actions/workflows/deploy_docs.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/deploy_docs.yml) [![Test Suite](https://github.com/project-vajra/veeksha/actions/workflows/test_suite.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/test_suite.yml) [![Run Linters](https://github.com/project-vajra/veeksha/actions/workflows/lint.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/lint.yml)

Veeksha is a LLM Inference systems benchmarking tool. Please refer to our [documentation](https://project-vajra.github.io/veeksha) and [paper](https://arxiv.org/abs/2407.07000) for more details.

## Setup

### Clone repository
```bash
git clone https://github.com/project-vajra/veeksha.git
cd veeksha
```

### Create uv environment and install the dependencies

Install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment and install Veeksha. We recommend using Python 3.14 free-threaded:
```bash
uv venv --python 3.14t
source .venv/bin/activate
uv pip install -e .
```

### Setup Wandb [Optional]
First create and setup your account at `https://<your-org>.wandb.io/` or public Wandb and obtain API key. Then run the following command and enter API key linked to your wandb account:
```bash
wandb login --host https://<your-org>.wandb.io
```
To opt out of wandb, do any of the following:
1. Don't pass any wandb related args like `--wandb-project`, `--wandb-group` and `wandb-run-name` when running python scripts. Alternatively, pass in `--no-should-write-metrics` instead of `--should-write-metrics` boolean flag.
2. Run `export WANDB_MODE=disabled` in your shell or add this to `~/.zshrc` or `~/.bashrc`. Remember to reload your shell using `source ~/.zshrc` or `source ~/.bashrc`.

## Running Code

### Running with Public APIs
#### Export API Key and URL
```bash
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE=https://api.endpoints.anyscale.com/v1
```
#### Running Benchmark
```bash
python -Xgil=0 -m veeksha.benchmark \
--client-config-model "Qwen/Qwen3-4B-Instruct-2507" \
--max-completed-requests 100 \
--timeout 600 \
--metrics-config-output-dir "benchmark_outputs" \
--synthetic-request-generator-config-interval-generator-config-type "static"
```

**Note:** We recommend running with `-Xgil=0` to enable GIL-free Python with true parallel execution of worker threads.

There are many more arguments for running benchmark, run the following to know more:
```bash
python -m veeksha.benchmark -h
```

### Running with Open Source Systems
veeksha can be run with any open source LLM inference system. If open source system does not provide OpenAI Compatible APIs, then kindly implement new LLM clients to support new open source system as explained in [here](#implementing-new-llm-clients).

Here we give an example with vLLM.

#### Launch vLLM Server
```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123 -tp 1 --rope-scaling '{"type":"dynamic","factor":2.0}'
```

If we need higher context length than supported by the model with certain scale factor, then we can add rope-scaling as `--rope-scaling '{"type":"dynamic","factor":2.0}'`. Adjust type and factor as per the use case.

#### Export API Key and URL
```bash
export OPENAI_API_KEY=token-abc123
export OPENAI_API_BASE=http://localhost:8000/v1
```

And then we can run the benchmark as shown [here](#running-benchmark). Be sure to update `--model` flag to same model used to launch vLLM.

### Saving Results

The results of the benchmark are saved in the results directory specified by the `--output-dir` argument.

## Running Microbenchmarks with Server Orchestration

Veeksha now supports automatic server lifecycle management for running microbenchmarks! This is especially useful when you have limited GPU resources and need to run many experiments.

### Quick Start

Run a benchmark with automatic vLLM server management:

```bash
python -m veeksha.orchestration.run_microbenchmark \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 1 \
    --max-completed-requests 50 \
    --output-dir ./results
```

Run a parameter sweep across different configurations:

```bash
python -m veeksha.orchestration.run_microbenchmark \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --sweep-tensor-parallel 1 2 4 \
    --max-completed-requests 50 \
    --output-dir ./results
```

The system will automatically:
1. Launch the server with your specified configuration
2. Wait for the server to become ready
3. Run the benchmark
4. Shutdown the server and free resources

### Features

- **Resource Awareness**: Specify GPU allocation, tensor parallelism, and resource constraints
- **Automatic Lifecycle**: Launch, health check, and shutdown servers automatically  
- **Parameter Sweeps**: Run benchmarks across multiple configurations efficiently
- **Multiple Systems**: Support for vLLM (more systems coming soon)

For detailed documentation, see [veeksha/orchestration/README.md](veeksha/orchestration/README.md)

## Running Capacity Search
Refer to [readme](veeksha/capacity_search/README.md) file of `veeksha/capacity_search` folder to know more about how to run capacity search.


## Citation
If you use our work, please consider citing our paper:
```cite
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
This repository was originally created as fork from [LLMPerf](https://github.com/ray-project/llmperf) project.

