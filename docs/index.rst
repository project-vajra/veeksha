.. veeksha documentation master file, created by
   sphinx-quickstart on Sat Jul  6 17:47:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Veeksha Documentation
=====================

**Veeksha** is a high-fidelity benchmarking framework for LLM inference systems.
Whether you're optimizing a production deployment, comparing serving backends, or
running capacity planning experiments, Veeksha gives you precise, reproducible
measurements with realistic workloads.

Built for accuracy: Veeksha models real-world traffic patterns-multi-turn
conversations, arrival rate distributions, and shared prefix caching-so your
benchmarks reflect actual production behavior.

.. note::

   Veeksha (वीक्षा) means "observation" or "investigation" in Sanskrit.


Key Features
------------

**Realistic Workload Modeling**
    - **DAG-Based Sessions**: Model multi-turn conversations as directed acyclic graphs with
      history inheritance, capturing real chat context accumulation
    - **Shared Prefix Testing**: Generate workloads with configurable prefix sharing to
      benchmark KV-cache efficiency
    - **Trace Replay**: Replay production traces (Claude Code, RAG, conversational) with
      preserved timing and token distributions

**Flexible Traffic Generation**
    - **Open-Loop (Rate-Based)**: Poisson, gamma, or fixed arrival rates to measure latency
      under realistic bursty traffic
    - **Closed-Loop (Concurrency-Based)**: Maintain target concurrent sessions with ramp-up
      control for throughput testing

**SLO-Aware Evaluation**
    - **Per-Request Metrics**: TTFC, TBC, TPOT, and end-to-end latency with percentile distributions
    - **Automated Health Checks**: Validates prompt/output lengths, arrival rates, and
      request dependencies to ensure benchmark correctness
    - **Capacity Search**: Adaptive probe-then-binary-search algorithm to find maximum
      sustainable throughput or rate meeting latency SLOs

**Production-Ready Tooling**
    - **Managed Server Orchestration**: Launch and manage vLLM/SGLang servers automatically
      with health checks and log capture
    - **Configuration Sweeps**: Use ``!expand`` YAML tag to run Cartesian product of
      parameter combinations with aggregated summaries
    - **WandB Integration**: Automatic logging of metrics, artifacts, and experiment tracking
      with sweep/capacity-search summaries


Quick Example
-------------

Run a simple benchmark against an OpenAI-compatible endpoint::

    python -Xgil=0 -m veeksha.benchmark \
        --client-type openai_chat_completions \
        --client-api-base http://localhost:8000/v1 \
        --client-model my-model \
        --traffic-scheduler-type rate \
        --traffic-scheduler-interval-generator-type poisson \
        --traffic-scheduler-interval-generator-arrival-rate 5.0 \
        --runtime-benchmark-timeout 60

Or use a YAML configuration file::

    python -Xgil=0 -m veeksha.benchmark --benchmark-config-from-file my_benchmark.veeksha.yml


Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   understanding_veeksha/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   basic_usage/index
   advanced_usage/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   config_reference/index
