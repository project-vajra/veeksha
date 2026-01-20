Quick start
===========

This guide walks you through running your first Veeksha benchmark in minutes.


Step 1: Create a configuration file
------------------------------------

Create a file named ``my_benchmark.veeksha.yml``:

.. code-block:: yaml

    # Basic benchmark configuration
    seed: 42

    # Where to send requests
    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct

    # Traffic pattern: 5 sessions per second
    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 5.0

    # Session content: single-turn with random prompts
    session_generator:
      type: synthetic
      session_graph:
        type: linear
        inherit_history: false
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

    # Stop conditions
    runtime:
      benchmark_timeout: 60     # Run for 60 seconds
      max_sessions: -1          # No limit on sessions

    # Enable metrics collection
    evaluators:
      - type: performance
        target_channels: ["text"]

.. tip::

   Use the ``.veeksha.yml`` extension for IDE autocompletion support. We highly recommend adding Veeksha's YAML schema to your IDE (see :ref:`Export JSON Schema <configuration-export-json-schema>`)


Step 2: Run the benchmark
-------------------------

Execute the benchmark using the CLI (against an already-running server; for managed servers see :doc:`/advanced_usage/server_management`):

.. code-block:: bash

    python -Xgil=0 -m veeksha.benchmark --benchmark-config-from-file my_benchmark.veeksha.yml

Step 3: View results
--------------------

Navigate to the output directory to find:

.. code-block:: text

    benchmark_output/09:01:2026-10:30:00-abc123/
    ├── config.yml                    # Resolved configuration
    ├── health_check_results.txt      # Benchmark verification
    ├── metrics/
    │   ├── request_level_metrics.jsonl   # Per-request data
    │   ├── summary_stats.json            # Aggregate statistics
    │   ├── ttfc.csv                      # TTFC percentiles
    │   ├── ttfc.png                      # TTFC distribution plot
    │   ├── tbc.csv                       # TBC percentiles  
    │   └── ...
    └── traces/
        └── trace.jsonl               # Request traces

Quick summary from ``summary_stats.json``:

.. code-block:: json

    {
      "Number of Requests": 287,
      "Number of Completed Requests": 287,
      "Error Rate": 0.0,
      "Observed Session Dispatch Rate": 4.78
    }


Essential configuration options
-------------------------------

Here are the most commonly adjusted options:

**Endpoint configuration:**

.. code-block:: yaml

    client:
      api_base: http://localhost:8000/v1  # Server URL
      model: meta-llama/Llama-3-8B-Instruct
      request_timeout: 300  # Timeout per request (seconds)

**Traffic rate:**

.. code-block:: yaml

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 10.0  # Sessions per second

**Benchmark duration:**

.. code-block:: yaml

    runtime:
      benchmark_timeout: 120  # Run for 2 minutes
      max_sessions: 500       # Or stop after 500 sessions (whichever first)
      # Use -1 for unlimited sessions

**Prompt/output lengths:**

.. code-block:: yaml

    session_generator:
      channels:
        - type: text
          body_length_generator:
            type: fixed
            value: 256       # Fixed 256 tokens per prompt
      output_spec:
        text:
          output_length_generator:
            type: fixed
            value: 128       # Request 128 token outputs


Using CLI overrides
-------------------

Override configuration values without editing the file:

.. code-block:: bash

    python -Xgil=0 -m veeksha.benchmark \
        --benchmark-config-from-file my_benchmark.veeksha.yml \
        --rate-traffic-scheduler-poisson-interval-generator-arrival-rate 20.0 \
        --runtime-benchmark-timeout 120

This runs at 20 sessions/second for 120 seconds instead of the file's values.


Common patterns
---------------

**Quick latency test at low load:**

.. code-block:: bash

    python -Xgil=0 -m veeksha.benchmark \
        --benchmark-config-from-file my_benchmark.veeksha.yml \
        --rate-traffic-scheduler-poisson-interval-generator-arrival-rate 1.0 \
        --runtime-max-sessions 50

**Throughput saturation test:**

.. code-block:: yaml

    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: 16
      rampup_seconds: 10

**Fixed prompt/output for consistent measurements:**

.. code-block:: yaml

    session_generator:
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


Configuration reference
-----------------------

For a detailed reference of all configuration options, see the :doc:`/config_reference/index`.

Next steps
----------

- :doc:`configuration` - Learn the full configuration system
- :doc:`output_files` - Understand all output files
- :doc:`/understanding_veeksha/scheduling` - Learn about traffic patterns
- :doc:`/advanced_usage/capacity_search` - Find maximum sustainable throughput
