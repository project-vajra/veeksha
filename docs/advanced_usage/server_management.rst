Server Management
=================

Veeksha can automatically launch and manage LLM inference servers, making
benchmarks fully self-contained and reproducible. This is especially useful
for CI pipelines and comparing different server configurations.


Supported servers
-----------------

Veeksha currently supports:

- **Vajra**
- **SGLang**
- **vLLM**


Basic configuration
-------------------

Add a ``server`` section to your benchmark config:

.. code-block:: yaml

    server:
      type: sglang            # or vllm, vajra
      env_path: sglang_env    # Python environment with server installed
      model: meta-llama/Llama-3-8B-Instruct
      host: localhost
      port: 30000

    # Client settings are automatically configured by the server
    client:
      type: openai_chat_completions
      request_timeout: 120

When ``server`` is configured:

1. Veeksha launches the server before the benchmark
2. Waits for the server to be healthy
3. Automatically sets ``client.api_base``, ``client.model``, and ``client.api_key``
4. Runs the benchmark
5. Shuts down the server when complete


Server configuration options
----------------------------

All server types share these common options:

.. code-block:: yaml

    server:
      type: sglang
      env_path: /path/to/sglang_env    # Python environment
      model: meta-llama/Llama-3-8B-Instruct

      # Network settings
      host: localhost
      port: 30000
      api_key: token-abc123            # Generated API key

      # GPU configuration
      gpu_ids: [0, 1]                  # Specific GPUs (null = auto-assign)
      tensor_parallel_size: 2          # Number of GPUs for TP
      require_contiguous_gpus: true    # Require consecutive GPU IDs

      # Model settings
      dtype: auto                      # float16, bfloat16, or auto
      max_model_len: 8192              # Maximum context length

      # Startup settings
      startup_timeout: 300             # Seconds to wait for server
      health_check_interval: 2.0       # Seconds between health checks

      # Additional server arguments
      additional_args: '{"enable_prefix_caching": true}'

``env_path``
    Path to a Python virtual environment or conda environment containing
    the server installation. Can be relative or absolute.

``gpu_ids``
    Explicit list of GPU IDs to use. If ``null``, GPUs are auto-assigned
    based on availability and ``tensor_parallel_size``.

``additional_args``
    JSON string or dict of extra arguments passed to the server command.


GPU resource management
-----------------------

Veeksha includes a resource manager for multi-GPU systems:

**Auto-assignment**

.. code-block:: yaml

    server:
      type: vllm
      tensor_parallel_size: 4
      gpu_ids: null             # Auto-assign 4 GPUs
      require_contiguous_gpus: true

The resource manager finds 4 contiguous available GPUs.

**Explicit assignment**

.. code-block:: yaml

    server:
      type: sglang
      tensor_parallel_size: 2
      gpu_ids: [2, 3]           # Use GPUs 2 and 3 specifically

**Non-contiguous GPUs** (when supported)

.. code-block:: yaml

    server:
      type: vllm
      tensor_parallel_size: 2
      cpu_ids: [0, 2]           # Use GPUs 0 and 2
      require_contiguous_gpus: false

Server logs
-----------

Server stdout/stderr are written to the benchmark output directory:

.. code-block:: text

    benchmark_output/09:01:2026-10:30:00-abc123/
    ├── server_logs_vajra_localhost_30003_20260109-110406.log
    └── ...

This is useful for debugging server issues.


Example: Full managed benchmark
-------------------------------

.. code-block:: yaml

    # managed_benchmark.veeksha.yml
    seed: 42
    output_dir: benchmark_output

    server:
      type: sglang
      env_path: ~/envs/sglang
      model: meta-llama/Llama-3-8B-Instruct
      host: localhost
      port: 30000
      tensor_parallel_size: 1
      max_model_len: 8192
      startup_timeout: 300

    client:
      type: openai_chat_completions
      request_timeout: 120
      max_tokens_param: max_tokens
      min_tokens_param: min_tokens

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 10.0

    session_generator:
      type: synthetic
      session_graph:
        type: linear
        inherit_history: true
      channels:
        - type: text
          body_length_generator:
            type: uniform
            min: 100
            max: 500
      output_spec:
        text:
          output_length_generator:
            type: uniform
            min: 100
            max: 300

    runtime:
      benchmark_timeout: 60
      max_sessions: -1

    evaluators:
      - type: performance
        target_channels: ["text"]


Example: Comparing servers
--------------------------

Create a base config and run with different servers:

.. code-block:: yaml

    # base_config.yml
    session_generator:
      type: synthetic
      session_graph:
        type: linear
      channels:
        - type: text
          body_length_generator:
            type: fixed
            value: 512
      output_spec:
        text:
          output_length_generator:
            type: fixed
            value: 256

    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: 8
      rampup_seconds: 5

    runtime:
      benchmark_timeout: 120

.. code-block:: bash

    # Run with vLLM
    python -Xgil=0 -m veeksha.benchmark \
        --benchmark-config-from-file base_config.yml \
        --server-type vllm \
        --vllm-server-env-path vllm_env \
        --vllm-server-model meta-llama/Llama-3.2-1B-Instruct \
        --output-dir results/vllm

    # Run with SGLang
    python -Xgil=0 -m veeksha.benchmark \
        --benchmark-config-from-file base_config.yml \
        --server-type sglang \
        --sglang-server-env-path sglang_env \
        --sglang-server-model meta-llama/Llama-3-8B-Instruct \
        --output-dir results/sglang
