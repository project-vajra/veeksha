Microbenchmarks
===============

Microbenchmarks isolate specific operations (prefill, decode) for precise
performance measurement. This guide covers how to configure Veeksha for
targeted microbenchmarking.


Prefill vs decode
-----------------

LLM inference has two main phases:

**Prefill (Prompt Processing)**
    Processing all input tokens to populate the KV cache.
    Compute-bound, scales with prompt length.

**Decode (Token Generation)**
    Generating output tokens one at a time.
    Memory-bandwidth bound, scales with batch size.

Measuring these separately helps identify bottlenecks.


Prefill microbenchmark
----------------------

Isolate prefill by using fixed-length prompts, minimal outputs, and no batching:

.. code-block:: yaml

    # prefill_benchmark.veeksha.yml
    seed: 42
    output_dir: microbench_prefill

    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: 1    # No batching
      rampup_seconds: 0

    session_generator:
      type: synthetic
      session_graph:
        type: linear
        inherit_history: false
        num_request_generator:
          type: fixed
          value: 1
      channels:
        - type: text
          body_length_generator:
            type: fixed_stair              # Sweep prompt lengths
            values: [128, 256, 512, 1024, 2048]
            repeat_each: 10                # 10 requests per length
      output_spec:
        text:
          output_length_generator:
            type: fixed
            value: 1                 # Minimal output

    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct

    runtime:
      max_sessions: 50               # 5 lengths × 10 samples
      benchmark_timeout: 600

    evaluators:
      - type: performance
        target_channels: ["text"]
        stream_metrics: false

The **stair length generator** creates requests with increasing prompt lengths:

.. code-block:: text

    10 requests at 128 tokens
    10 requests at 256 tokens
    10 requests at 512 tokens
    10 requests at 1024 tokens
    10 requests at 2048 tokens

Analyze ``prefill_stats.json`` to see TTFC grouped by prompt length.


Decode microbenchmark
---------------------

Isolate decode by using fixed prompts, variable outputs, and controlled batching:

.. code-block:: yaml

    # decode_benchmark.veeksha.yml
    seed: 42
    output_dir: microbench_decode

    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: 8   # Batch size
      rampup_seconds: 0

    session_generator:
      type: synthetic
      session_graph:
        type: linear
        inherit_history: false
        num_request_generator:
          type: fixed
          value: 1
      channels:
        - type: text
          body_length_generator:
            type: fixed
            value: 128              # Minimal context
      output_spec:
        text:
          output_length_generator:
            type: fixed
            value: 512              # Focus on decode

    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct
      min_tokens_param: min_tokens  # Ensure exact output length

    runtime:
      max_sessions: 100
      benchmark_timeout: 600

    evaluators:
      - type: performance
        target_channels: ["text"]
        stream_metrics: false
        text_channel:
          decode_window_enabled: true
          decode_window_config:
            min_active_requests: 8         # Only measure at full batch
            selection_strategy: all
            anchor_to_client_pickup: true
            require_streaming: true


Decode window analysis
----------------------

The **decode window** isolates steady-state decode performance by filtering
out prefill and ramp-up effects:

.. code-block:: yaml

    text_channel:
      decode_window_enabled: true
      decode_window_config:
        min_active_requests: 8        # Minimum concurrent requests
        selection_strategy: all       # Include all tokens in window
        anchor_to_client_pickup: true # Start from first response
        require_streaming: true       # Only streaming responses

``min_active_requests``
    Only measure tokens when at least this many requests are active.
    Use ``"max_observed"`` to auto-detect peak batch size.

``selection_strategy``
    - ``all``: Include all tokens in the window
    - ``middle``: Exclude first/last portions

This produces accurate per-token decode throughput without prefill influence.


Decode window output files
~~~~~~~~~~~~~~~~~~~~~~~~~~

When decode window analysis is enabled, additional files are generated in the
``metrics/`` directory:

**decode_window_metrics.json**
    Detailed decode window statistics including per-batch TBC measurements
    and throughput calculations.

**decode_window_plot.png**
    Visualization showing token generation over time with the decode window
    highlighted:

.. image:: /_static/assets/decode_window_plot.png
   :alt: Decode window analysis plot
   :width: 600px

The plot shows active requests over time, with the decode windows
(steady-state regions) shaded. Only tokens within these windows contribute
to the final, windowed, decode throughput metrics.


Batch size sweep
----------------

Measure decode throughput across batch sizes:

.. code-block:: yaml

    # decode_batch_sweep.veeksha.yml
    seed: 42
    output_dir: microbench_decode_batch

    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: !expand [1, 2, 4, 8, 16, 32]
      rampup_seconds: 0

    session_generator:
      type: synthetic
      session_graph:
        type: linear
        num_request_generator:
          type: fixed
          value: 1
      channels:
        - type: text
          body_length_generator:
            type: fixed
            value: 128
      output_spec:
        text:
          output_length_generator:
            type: fixed
            value: 256

    evaluators:
      - type: performance
        text_channel:
          decode_window_enabled: true
          decode_window_config:
            min_active_requests: "max_observed"

Run and compare ``throughput_metrics.json`` across runs.
