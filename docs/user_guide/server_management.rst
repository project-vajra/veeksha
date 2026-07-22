Server Management
=================

Veeksha can automatically launch and manage LLM inference servers, making
benchmarks fully self-contained and reproducible. This is especially useful
for CI pipelines and comparing different server configurations.


Supported servers
-----------------

Veeksha currently supports:

- **vLLM Omni** in Docker
- **SGLang Omni** in Docker
- **Vajra** as a subprocess from a separate source checkout and environment


Basic configuration
-------------------

Add a ``server`` section to your benchmark config:

.. code-block:: yaml

    server:
      type: vllm
      image: vllm-omni:0.21-local
      hf_model: meta-llama/Llama-3.2-1B-Instruct
      deploy_config: /absolute/path/to/vllm_omni_deploy.yaml
      bootstrap: ""
      docker_gpus: all
      host: localhost
      port: 30000

    # Model, API base, and key are supplied by the managed endpoint.
    client:
      type: openai_chat_completions
      request_timeout: 120

When ``server`` is configured, Veeksha starts it, waits for health, applies
``client.api_base``, ``client.model``, and ``client.api_key``, runs the
benchmark, and always shuts the server down.

For a ``!expand`` sweep, resolved benchmark configs are grouped by their
complete ``server`` config. One server is reused by every run in a group.
Expanding any server field creates a separate group and lifecycle.


Server configuration options
----------------------------

All server types share network, endpoint, startup, and GPU allocation fields:

.. code-block:: yaml

    server:
      type: vllm
      model: served-model
      host: localhost
      port: 30000
      api_key: token-abc123
      gpu_ids: [0, 1]
      tensor_parallel_size: 2
      require_contiguous_gpus: true
      startup_timeout: 300
      health_check_interval: 2.0
      health_url: null

``gpu_ids``
    Explicit host GPU IDs. If omitted, Veeksha allocates
    ``tensor_parallel_size`` GPUs. Docker configs can instead set
    ``docker_gpus`` to an argument accepted by ``docker run --gpus``, such as
    ``all`` or ``device=0,1``.

``model``
    Model name written into the endpoint and applied to the client. It
    defaults to ``hf_model`` for vLLM and to ``model_name`` or ``model_path``
    for SGLang.

vLLM Omni container
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    server:
      type: vllm
      image: vllm-omni:0.21-local
      hf_model: meta-llama/Llama-3.2-1B-Instruct
      deploy_config: /absolute/path/to/vllm_omni_deploy.yaml
      container_deploy_config: /etc/vllm-omni/deploy.yaml  # optional
      docker_gpus: all
      engine_args:
        - --trust-remote-code
      env:
        HF_TOKEN: token-value
      pass_env: []
      volumes: []

``deploy_config`` must be a host file. Veeksha mounts it read-only into the
container and passes the container path to ``vllm serve --deploy-config``.

SGLang Omni container
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    server:
      type: sglang
      image: frankleeeee/sglang-omni:dev
      model_path: Qwen/Qwen2.5-Omni-7B
      model_name: qwen-omni
      deploy_config: /absolute/path/to/sglang_omni_deploy.yaml
      source_dir: /absolute/path/to/sglang-omni
      docker_gpus: all
      shm_size: 32g

The default SGLang bootstrap mounts ``source_dir``, creates a container-local
virtual environment, and installs SGLang Omni before serving. For an image that
already contains ``sgl-omni``, set ``bootstrap: ""``; then ``source_dir`` is
not required.

Vajra subprocess
~~~~~~~~~~~~~~~~

.. code-block:: yaml

    server:
      type: vajra
      model: served-model
      setup_dir: /absolute/path/to/vajra
      command:
        - /absolute/path/to/vajra/env/bin/python
        - -m
        - vajra.entrypoints.api_server.server
        - --port
        - "30000"

The interpreter must be the Vajra environment's absolute Python path. Veeksha
records the source checkout's current Git commit with the benchmark artifacts.


GPU resource management
-----------------------

Veeksha includes a resource manager for multi-GPU systems. The snippets below
show only allocation fields; combine them with the required engine fields from
the container examples above.

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
      gpu_ids: [0, 2]           # Use GPUs 0 and 2
      require_contiguous_gpus: false

Server logs
-----------

Container logs and reproducibility details are written under the output
directory used for that lifecycle:

.. code-block:: text

    benchmark_output/sweep_.../
    ├── managed_server_01/
    │   ├── engine_details.json
    │   └── vllm_docker_1.log
    ├── <benchmark-run>/
    └── ...

``engine_details.json`` records the image, image hash, container ID, and name.


Example: Full managed benchmark
-------------------------------

.. code-block:: yaml

    # managed_benchmark.veeksha.yml
    seed: 42
    output_dir: benchmark_output

    server:
      type: vllm
      image: vllm-omni:0.21-local
      hf_model: meta-llama/Llama-3.2-1B-Instruct
      deploy_config: /absolute/path/to/vllm_omni_deploy.yaml
      bootstrap: ""
      docker_gpus: all
      host: localhost
      port: 30000
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

    # Run with vLLM Omni
    uvx -p 3.14t veeksha benchmark \
        --config base_config.yml \
        --server.type vllm \
        --server.image vllm-omni:0.21-local \
        --server.hf_model meta-llama/Llama-3.2-1B-Instruct \
        --server.deploy_config /path/to/vllm_deploy.yaml \
        --server.docker_gpus all \
        --output_dir results/vllm

    # Run with a pre-baked SGLang Omni image
    uvx -p 3.14t veeksha benchmark \
        --config base_config.yml \
        --server.type sglang \
        --server.model_path Qwen/Qwen2.5-Omni-7B \
        --server.deploy_config /path/to/sglang_deploy.yaml \
        --server.bootstrap "" \
        --server.docker_gpus all \
        --output_dir results/sglang
