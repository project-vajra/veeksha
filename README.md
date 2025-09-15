# Veeksha

Veeksha is a LLM Inference systems benchmarking tool. Please refer to our [documentation](https://project-vajra.github.io/veeksha) and [paper](https://arxiv.org/abs/2407.07000) for more details.

## Setup

### Clone repository
```bash
git clone https://github.com/project-vajra/veeksha.git
cd veeksha
```

### Create conda environment
```bash
conda create -p ./env python=3.12
conda activate ./env
```

### Install veeksha
```bash
pip install -e .
```

### Setup Wandb [Optional]
First create and setup your account at `https://wandb.ai/` (or your private instance at `https://<your-org>.wandb.io/`) and obtain API key. Then run the following command and enter API key linked to your wandb account:

For public Wandb:
```bash
wandb login
```

For private Wandb instance:
```bash
wandb login --host https://<your-org>.wandb.io
```

#### Using Wandb with Veeksha
To enable wandb logging when running benchmarks, add these parameters:
```bash
--metrics-config-should-write-metrics-to-wandb \
--metrics-config-wandb-project "YourProject" \
--metrics-config-wandb-group "YourGroup" \
--metrics-config-wandb-run-name "YourRun"
```

#### Disabling Wandb
To opt out of wandb, do any of the following:
1. Don't pass any wandb related args like `--metrics-config-wandb-project`, `--metrics-config-wandb-group` and `--metrics-config-wandb-run-name` when running python scripts. Alternatively, pass in `--no-metrics-config-should-write-metrics-to-wandb` instead of `--metrics-config-should-write-metrics-to-wandb` boolean flag.
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
python -m veeksha.benchmark \
--client-config-model "meta-llama/Meta-Llama-3-8B-Instruct" \
--max-completed-requests 150 \
--timeout 600 \
--client-config-num-clients 2 \
--client-config-num-concurrent-requests-per-client 5 \
--metrics-config-output-dir "result_outputs" \
--request-generator-config-type "synthetic" \
--synthetic-request-generator-config-interval-generator-config-type "poisson" \
--synthetic-request-generator-config-poisson-interval-generator-config-qps 0.5 \
--synthetic-request-generator-config-length-generator-config-type "trace" \
--synthetic-request-generator-config-trace-length-generator-config-trace-file "veeksha/data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv" \
--synthetic-request-generator-config-trace-length-generator-config-max-tokens 8192 \
--metrics-config-deadline-report-ttft-deadline 0.3 \
--metrics-config-deadline-report-tbt-deadline 0.03 \
--metrics-config-should-write-metrics-to-wandb \
--metrics-config-wandb-project Project \
--metrics-config-wandb-group Group \
--metrics-config-wandb-run-name Run
```

There are many more arguments for running benchmark, run the following to know more:
```bash
python -m veeksha.benchmark -h
```

### Running with Open Source Systems
veeksha can be run with any open source LLM inference system. If open source system does not provide OpenAI Compatible APIs, then kindly implement new LLM clients to support new open source system as explained in [here](#implementing-new-llm-clients).

Here we give an example with vLLM.

#### Launch vLLM Server
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key token-abc123 \
  -tp 1 \
  --rope-scaling '{"rope_type":"dynamic","factor":2.0}'

```

If we need higher context length than supported by the model with certain scale factor, then we can add rope-scaling as `--rope-scaling '{"rope_type":"dynamic","factor":2.0}'`. Adjust type and factor as per the use case.

#### Export API Key and URL
```bash
export OPENAI_API_KEY=token-abc123
export OPENAI_API_BASE=http://localhost:8000/v1
```

And then we can run the benchmark as shown [here](#running-benchmark). Be sure to update `--client-config-model` flag to same model used to launch vLLM.

### Saving Results

The results of the benchmark are saved in the results directory specified by the `--metrics-config-output-dir` argument.

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

