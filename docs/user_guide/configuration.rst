Configuration System
====================

Veeksha uses a flexible polymorphic configuration system that supports YAML files,
CLI arguments, and programmatic access. This guide explains how the system works
and how to navigate it effectively.


Configuration methods
---------------------

**YAML Files** (recommended)
    Create a ``.veeksha.yml`` file with your configuration:

    .. code-block:: yaml

        seed: 42
        client:
          type: openai_chat_completions
          api_base: http://localhost:8000/v1
          model: my-model
        traffic_scheduler:
          type: rate
          interval_generator:
            type: poisson
            arrival_rate: 10.0

**CLI Arguments**
    Override any option using dot notation:

    .. code-block:: bash

        uvx veeksha benchmark \
            --client.api_base http://localhost:8000/v1 \
            --traffic_scheduler.interval_generator.arrival_rate 20.0

Argument names mirror the YAML hierarchy with dots.

**Combined** (YAML + CLI)
    CLI arguments override YAML values:

    .. code-block:: bash

        # Base config from file, override arrival rate
        uvx veeksha benchmark \
            --config base.veeksha.yml \
            --traffic_scheduler.interval_generator.arrival_rate 30.0


Polymorphic options
-------------------

Many options have a ``type`` field that selects a variant with its own options:

.. code-block:: yaml

    # Session generator can be: synthetic, trace, or lmeval
    session_generator:
      type: synthetic        # Selects synthetic variant
      session_graph:         # Options specific to synthetic
        type: linear
      channels:
        - type: text

    # Traffic scheduler can be: rate or concurrent
    traffic_scheduler:
      type: rate             # Selects rate variant
      interval_generator:    # Options specific to rate
        type: poisson
        arrival_rate: 10.0

Each ``type`` exposes different options. See the :doc:`/config_reference/index` for the full list.

.. _configuration-export-json-schema:

Exporting JSON schema
---------------------

Export a JSON schema for YAML IDE autocompletion and linting:

.. code-block:: bash

    uvx veeksha benchmark --export-json-schema veeksha-schema.json

Configure your IDE to use this schema. In VSCode and forks:

.. code-block:: json

    // .vscode/settings.json
    {
        "yaml.schemas": {
            "./veeksha-schema.json": "*.veeksha.yml"
        },
        "yaml.customTags": [
            "!expand sequence"
        ]
    }

.. hint::
  The YAML IDE extension may be required for "yaml.schemas" to show up as a valid setting.

.. figure:: /_static/assets/yaml_help_text.png
   :alt: VSCode YAML integration example
   :align: center
   :width: 600px

   The VSCode YAML extension providing autocompletion and documentation on hover.


Common configuration sections
-----------------------------

**client** - API endpoint configuration

.. code-block:: yaml

    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct
      # api_key: optional, falls back to OPENAI_API_KEY env var
      request_timeout: 300
      max_tokens_param: max_completion_tokens
      min_tokens_param: min_tokens

**traffic_scheduler** - Traffic pattern

.. code-block:: yaml

    # Rate-based
    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 10.0
      cancel_session_on_failure: true

    # OR Concurrency-based
    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: 8
      rampup_seconds: 10

**session_generator** - Content generation

.. code-block:: yaml

    session_generator:
      type: synthetic
      session_graph:
        type: linear
        num_request_generator:
          type: uniform
          min: 1
          max: 5
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
            min: 50
            max: 200

**runtime** - Execution parameters

.. code-block:: yaml

    runtime:
      benchmark_timeout: 300      # Total benchmark duration
      max_sessions: 1000          # Maximum sessions (-1 = unlimited)
      post_timeout_grace_seconds: 10  # Wait for in-flight after timeout
      num_client_threads: 3       # Async HTTP client threads

**evaluators** - Metrics collection

.. code-block:: yaml

    evaluators:
      - type: performance
        target_channels: ["text"]
        stream_metrics: true
        slos:
          - name: "P99 TTFC"
            metric: ttfc
            percentile: 0.99
            value: 0.5
            type: constant


Environment variables
---------------------

Veeksha automatically reads certain environment variables as fallbacks when
configuration values are not explicitly set:

``OPENAI_API_KEY``
    Used as the API key if ``client.api_key`` is not set in config.

``OPENAI_API_BASE``
    Used as the API base URL if ``client.api_base`` is not set in config.

This allows you to set credentials once in your environment:

.. code-block:: bash

    export OPENAI_API_KEY=your-api-key
    export OPENAI_API_BASE=http://localhost:8000/v1

Then omit them from your config file:

.. code-block:: yaml

    # No need to specify api_key or api_base
    client:
      type: openai_chat_completions
      model: meta-llama/Llama-3-8B-Instruct

This is especially useful for:

- Avoiding committing secrets to version control
- Sharing configs across environments with different servers

Veeksha also reads ``HF_TOKEN`` from the environment in order to access gated models.

Stop conditions
---------------

Benchmarks stop when either condition is met:

.. code-block:: yaml

    runtime:
      benchmark_timeout: 300    # Stop after 300 seconds
      max_sessions: 1000        # OR after 1000 sessions

Use ``-1`` for unlimited:

.. code-block:: yaml

    runtime:
      benchmark_timeout: -1     # Run indefinitely
      max_sessions: 500         # Stop only after 500 sessions

When a timeout hits, Veeksha will record all in-flight requests and keep dispatching sessions as usual. 
Then, it will exit after ``post_timeout_grace_seconds`` have passed, only if the session limit is not reached before that. 

.. code-block:: yaml

    runtime:
      benchmark_timeout: 60
      post_timeout_grace_seconds: 10  # Wait 10s for in-flight requests
      # -1 = wait indefinitely for all in-flight
      # 0 = exit immediately (cancel in-flight)


Output directory
----------------

Control where results are saved:

.. code-block:: yaml

    output_dir: benchmark_output

Results are saved to a timestamped subdirectory:

.. code-block:: text

    benchmark_output/
    └── 09:01:2026-10:30:00-a1b2c3d4/
        ├── config.yml
        ├── metrics/
        └── traces/

The subdirectory name includes:

- Date and time
- Short hash of the configuration (for uniqueness)


Trace recording
---------------

Control what's recorded for debugging:

.. code-block:: yaml

    trace_recorder:
      enabled: true          # Write trace file
      include_content: false # Exclude prompt/response content (smaller files)

Set ``include_content: true`` to record full request content for debugging.


Validation
----------

Veeksha validates configurations at startup:

- Type checking for all fields
- Enum validation for ``type`` fields
- Required field checking
- Cross-field validation (e.g., ``min <= max``)

Invalid configurations produce clear error messages:

.. code-block:: text

    ConfigurationError: traffic_scheduler.interval_generator.arrival_rate
    must be positive, got -5.0


.. _configuration-splitting:

Splitting configuration across files
-------------------------------------

For better organization and reusability, you can split your configuration across
multiple YAML files using the ``!include`` tag. This is useful when you want to:

- Reuse client configuration across different benchmarks
- Keep environment-specific settings (e.g., API endpoints) separate
- Share traffic patterns across experiments

**Example: Separate client and traffic configs**

Create ``client.yml`` with just client settings:

.. code-block:: yaml

    # client.yml
    type: openai_chat_completions
    api_base: http://localhost:8000/v1
    model: meta-llama/Llama-3-8B-Instruct

Create ``traffic.yml`` with traffic settings:

.. code-block:: yaml

    # traffic.yml
    type: rate
    interval_generator:
      type: poisson
      arrival_rate: 5.0

Create a main config that includes both:

.. code-block:: yaml

    # main_config.yml
    seed: 42

    client: !include client.yml
    traffic_scheduler: !include traffic.yml

    session_generator:
      type: synthetic
      channels:
        - type: text
          body_length_generator:
            type: uniform
            min: 50
            max: 200

    runtime:
      benchmark_timeout: 60

Run the benchmark with a single ``--config`` flag:

.. code-block:: bash

    uvx veeksha benchmark --config main_config.yml

**CLI overrides still work**

You can override any value from the included files using CLI arguments:

.. code-block:: bash

    uvx veeksha benchmark \
        --config main_config.yml \
        --client.model llama-70b  # Override model from client.yml


.. _workload-recipes:

Workload recipes
================

This section shows complete, runnable configurations for common benchmarking
scenarios. Each recipe is a standalone YAML file — copy it, point ``api_base``
at your server, and run.


Replay a request log (CSV or JSONL)
------------------------------------

Replay a simple trace of independent requests with just token length
distributions. Works with CSV files directly — no conversion needed:

.. code-block:: yaml

    # trace_request_log.veeksha.yml
    seed: 42

    session_generator:
      type: trace
      trace_file: sharegpt_8k_filtered.csv   # CSV or JSONL
      wrap_mode: true
      flavor:
        type: request_log

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 10.0

    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct

    runtime:
      benchmark_timeout: 120
      max_sessions: -1

    evaluators:
      - type: performance
        target_channels: ["text"]

The trace file needs ``input_length`` and ``output_length`` columns (or the
common alternatives ``num_prefill_tokens`` / ``num_decode_tokens``, which are
auto-normalized). Each row becomes an independent single-request session.

**Trace flavors:**

``request_log``
    Independent requests with just token lengths. No session structure,
    no corpus files. Supports CSV and JSONL. Best for replaying public
    benchmarking datasets (ShareGPT, etc.).

``timed_synthetic_session``
    Timed session traces with synthetic content. Supports DAG replay through
    ``session_context`` and context caching via ``page_size``. Best for testing
    KV-cache reuse across linear and non-linear sessions.

``untimed_content_multi_turn``
    Replay conversation datasets with actual message content (ShareGPT,
    LMSYS-Chat, etc.). Configurable message schema for different dataset
    formats.

``shared_prefix``
    Multi-turn conversation dataset. Uses hash-based deterministic content
    generation with configurable ``block_size``.

``rag``
    Single-turn retrieval-augmented generation. Includes ``num_documents``
    warmup documents. Good for testing long-context prefill.


Replay conversation datasets
------------------------------

Replay datasets with actual conversation content (e.g. ShareGPT):

.. code-block:: yaml

    # conversation_replay.veeksha.yml
    seed: 42

    session_generator:
      type: trace
      trace_file: sharegpt_52k.jsonl
      wrap_mode: true
      flavor:
        type: untimed_content_multi_turn
        conversation_column: conversations
        role_key: from
        content_key: value
        user_role_value: human
        assistant_role_value: gpt

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 10.0

    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct

    runtime:
      benchmark_timeout: 120
      max_sessions: -1

    evaluators:
      - type: performance
        target_channels: ["text"]

Each row in the trace file should contain a ``conversations`` column (configurable)
with a list of message dicts. The schema keys (``role_key``, ``content_key``,
``user_role_value``, ``assistant_role_value``) can be customized for different
dataset formats. For LMSYS-Chat format, use ``role_key: role``,
``content_key: content``, ``user_role_value: user``,
``assistant_role_value: assistant``.


Replay timed multi-turn traces
-------------------------------

Replay timed multi-turn coding assistant traces with context caching:

.. code-block:: yaml

    # trace_timed_synthetic_session.veeksha.yml
    seed: 42

    session_generator:
      type: trace
      trace_file: traces/timed_synthetic_trace.jsonl
      wrap_mode: true
      flavor:
        type: timed_synthetic_session
        corpus_file: traces/corpus.txt
        page_size: 16

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 5.0

    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct
      request_timeout: 120
      max_tokens_param: max_completion_tokens

    runtime:
      benchmark_timeout: 120
      max_sessions: -1


Replay shared-prefix traces
-----------------------------

.. code-block:: yaml

    # trace_shared_prefix.veeksha.yml
    session_generator:
      type: trace
      trace_file: traces/shared_prefix_trace.jsonl
      flavor:
        type: shared_prefix
        corpus_file: traces/corpus.txt
        block_size: 512

    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: 32
      rampup_seconds: 10

    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct

    runtime:
      benchmark_timeout: 300
      max_sessions: -1


Multi-turn conversations (synthetic)
-------------------------------------

Generate multi-turn sessions with history accumulation and shared prefixes:

.. code-block:: yaml

    # multi_turn.veeksha.yml
    seed: 42

    session_generator:
      type: synthetic
      session_graph:
        type: linear
        inherit_history: true       # Each turn includes prior turns as context
        num_request_generator:
          type: uniform
          min: 2
          max: 4                    # 2-4 turns per session
        request_wait_generator:
          type: poisson
          arrival_rate: 5           # ~200ms think time between turns
      channels:
        - type: text
          shared_prefix_ratio: 0.2           # 20% of prompt is shared
          shared_prefix_probability: 0.5     # 50% of sessions share a prefix
          body_length_generator:
            type: uniform
            min: 100
            max: 500
      output_spec:
        text:
          output_length_generator:
            type: uniform
            min: 50
            max: 200

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 10.0

    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct

    runtime:
      benchmark_timeout: 60
      max_sessions: -1

    evaluators:
      - type: performance
        target_channels: ["text"]
        slos:
          - name: "P99 TTFC under 500ms"
            metric: ttfc
            percentile: 0.99
            value: 0.5
            type: constant


Agentic workloads (branching sessions)
--------------------------------------

Simulate agentic tool-calling patterns with fan-out/fan-in DAG structure:

.. code-block:: yaml

    # agentic.veeksha.yml
    seed: 42

    session_generator:
      type: synthetic
      session_graph:
        type: branching
        num_layers_generator:
          type: uniform
          min: 3
          max: 5
        layer_width_generator:
          type: uniform
          min: 2
          max: 6
        fan_out_generator:
          type: uniform
          min: 1
          max: 5
        fan_in_generator:
          type: uniform
          min: 1
          max: 4
        connection_dist_generator:
          type: uniform
          min: 1
          max: 2          # Allow skip connections
        single_root: true
        inherit_history: true
        request_wait_generator:
          type: poisson
          arrival_rate: 3
      channels:
        - type: text
          body_length_generator:
            type: uniform
            min: 50
            max: 200
      output_spec:
        text:
          output_length_generator:
            type: uniform
            min: 100
            max: 300

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 5.0

    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct

    runtime:
      max_sessions: 100
      benchmark_timeout: 120

    evaluators:
      - type: performance
        target_channels: ["text"]


LM-Eval accuracy benchmarks
----------------------------

Run standardized evaluation tasks from the `lm-evaluation-harness
<https://github.com/EleutherAI/lm-evaluation-harness>`_:

.. code-block:: yaml

    # lmeval.veeksha.yml
    seed: 42

    session_generator:
      type: lmeval
      tasks: ["triviaqa", "truthfulqa_gen"]
      num_fewshot: 0

    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: 4
      rampup_seconds: 0
      cancel_session_on_failure: false

    evaluators:
      - type: performance
        target_channels: ["text"]
      - type: accuracy_lmeval
        bootstrap_iters: 200

    client:
      type: openai_completions            # Note: completions, not chat
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct
      request_timeout: 240
      max_tokens_param: max_tokens
      additional_sampling_params: '{"temperature": 0}'

    runtime:
      max_sessions: 40
      benchmark_timeout: 1200

.. note::

   LM-Eval uses ``openai_completions`` (not ``openai_chat_completions``) for
   generation tasks. The ``accuracy_lmeval`` evaluator computes task-specific
   metrics alongside the standard performance evaluator.


Throughput saturation test
--------------------------

Push the server to maximum throughput using closed-loop concurrency:

.. code-block:: yaml

    # throughput.veeksha.yml
    seed: 42

    session_generator:
      type: synthetic
      session_graph:
        type: single_request
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
      target_concurrent_sessions: 32
      rampup_seconds: 10

    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct

    runtime:
      benchmark_timeout: 120
      max_sessions: -1

    evaluators:
      - type: performance
        target_channels: ["text"]


See also
--------

- :doc:`/config_reference/api_reference/BenchmarkConfig` - Complete benchmark configuration reference
- :doc:`/config_reference/api_reference/CapacitySearchConfig` - Capacity search configuration reference
