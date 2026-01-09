# Veeksha

[![Publish Release to PyPI](https://github.com/project-vajra/veeksha/actions/workflows/publish_release.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/publish_release.yml) [![Deploy Documentation](https://github.com/project-vajra/veeksha/actions/workflows/deploy_docs.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/deploy_docs.yml) [![Test Suite](https://github.com/project-vajra/veeksha/actions/workflows/test_suite.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/test_veeksha.yml) [![Run Linters](https://github.com/project-vajra/veeksha/actions/workflows/lint.yml/badge.svg)](https://github.com/project-vajra/veeksha/actions/workflows/lint.yml) 

Veeksha is a framework for evaluating the performance of LLM inference systems.  It allows users to simulate almost any real-life workload at scale by easily shaping the content, scheduling and evaluation schemas.

- Content: Veeksha's content primitive is a session (a Directed Acyclic Graph of requests). From single-turn text generation tasks to multi-turn, non-linear agentic patterns with content inheritance, sessions can represent any workload shape. Moreover, the content in each request can be customized for any modality (multi-modality coming soon).

- Scheduling: Veeksha supports synchronous intra-session dependencies, while asynchronously scheduling multiple sessions to simulate real-world usage patterns. One can use Veeksha to simulate a variety of session traffic, from a set concurrency level to a target rate of dispatch.

- Evaluation: Veeksha provides automatic evaluation of the performance metrics one would expect (throughput, latency, time to first token, time between tokens, etc.) as well as accuracy metrics for supported content (we support LM-Eval content and evaluation out of the box). All with native Weights & Biases integration, so that you can always track and visualize your results.

Please refer to our [documentation](https://project-vajra.github.io/veeksha) for more.

## Setup

### From source

#### Clone repository
```bash
git clone https://github.com/project-vajra/veeksha.git
cd veeksha
```

#### Create uv environment and install the dependencies

Install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment and install Veeksha. We recommend using Python 3.14 free-threaded for true worker parallelism:

```bash
uv venv --python 3.14t
source .venv/bin/activate
uv pip install -e .
```


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
