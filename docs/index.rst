.. veeksha documentation master file, created by
   sphinx-quickstart on Sat Jul  6 17:47:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Veeksha Documentation
=====================

**Veeksha** is a high-fidelity benchmarking framework for LLM inference systems.
Whether you're optimizing a production deployment, comparing serving backends, or
running capacity planning experiments, Veeksha lets you measure what matters to you:
realistic multi-turn conversations, agentic workflows, high-frequency stress tests, or targeted
microbenchmarks. One tool, any workload.

**From isolated requests to complex agentic sessions, Veeksha captures the full complexity of modern LLM workloads.**

👉 **New here?** Start with :doc:`why_veeksha` to learn what
sets Veeksha apart.

.. note::

   Veeksha (वीक्षा) means "observation" or "investigation" in Sanskrit.


Key features
------------

**Realistic workload modeling**
    - **DAG-based sessions**: Model multi-turn conversations and complex agentic workflows 
      as directed acyclic graphs with history inheritance, capturing real chat context accumulation
    - **Shared prefix testing**: Generate workloads with configurable prefix sharing to
      benchmark KV-cache efficiency
    - **Trace replay**: Replay production traces (Claude Code, RAG, conversational) with
      preserved timing and token distributions

**Flexible traffic generation**
    - **Open-loop (rate-based)**: Poisson, gamma, or fixed arrival rates to measure latency
      under realistic bursty traffic
    - **Closed-loop (concurrency-based)**: Maintain target concurrent sessions with ramp-up
      control for throughput testing

**SLO-aware evaluation**
    - **Per-request metrics**: TTFC, TBC, TPOT, and end-to-end latency with percentile distributions
    - **Automated health checks**: Validates prompt/output lengths, arrival rates, and
      request dependencies to ensure benchmark correctness
    - **Capacity search**: Adaptive probe-then-binary-search algorithm to find maximum
      sustainable throughput or rate meeting latency SLOs

**Production-ready tooling**
    - **Managed server orchestration**: Launch and manage inference servers automatically
      with health checks and log capture
    - **Configuration sweeps**: Use ``!expand`` YAML tag to run Cartesian product of
      parameter combinations with aggregated summaries
    - **WandB integration**: Automatic logging of metrics, artifacts, and experiment tracking
      with sweep/capacity-search summaries


Quick example
-------------

Run a simple benchmark against a running OpenAI-compatible endpoint::

    python -Xgil=0 -m veeksha.benchmark \
        --client-type openai_chat_completions \
        --openai-chat-completions-client-api-base http://localhost:8000/v1 \
        --openai-chat-completions-client-model meta-llama/Llama-3.2-1B-Instruct \
        --traffic-scheduler-type rate \
        --rate-traffic-scheduler-interval-generator-type poisson \
        --rate-traffic-scheduler-poisson-interval-generator-arrival-rate 2.0 \
        --runtime-benchmark-timeout 30

Or use a YAML configuration file::

    python -Xgil=0 -m veeksha.benchmark --benchmark-config-from-file my_benchmark.veeksha.yml


Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   why_veeksha
   installation

.. toctree::
   :maxdepth: 2
   :caption: Core concepts

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
