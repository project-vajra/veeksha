Capacity Search
===============

Capacity search automatically finds the maximum sustainable session rate or
concurrency that meets your latency service level objectives (SLOs). This is
essential for capacity planning and performance regression testing.


How it works
------------

Veeksha uses an **adaptive two-phase algorithm**:

**Phase 1: Exponential Probing**
    Start at a low value and exponentially increase until SLOs are violated.
    This quickly finds the approximate capacity ceiling.

**Phase 2: Binary Search**
    Perform binary search between the last passing and first failing values
    to converge on the precise capacity.

.. code-block:: text

    Example: Finding max rate

    Phase 1 (Probe):
      Rate 5.0 → PASS
      Rate 10.0 → PASS
      Rate 20.0 → PASS
      Rate 40.0 → FAIL  ← ceiling found

    Phase 2 (Binary):
      Rate 30.0 → PASS
      Rate 35.0 → PASS
      Rate 37.5 → FAIL
      Rate 36.25 → PASS
      → Converged at 36.25


Running capacity search
-----------------------

Create a capacity search configuration:

.. code-block:: yaml

    # capacity_search.veeksha.yml
    output_dir: capacity_search_output

    # Search parameters
    start_value: 5.0        # Initial probe value
    max_value: 100.0        # Maximum to search
    expansion_factor: 2.0   # Multiply by this during probing
    max_iterations: 20      # Maximum iterations
    precision: 2            # Decimal places for rate

    # Benchmark configuration (used for each iteration)
    benchmark_config:
      seed: 42

      traffic_scheduler:
        type: rate
        interval_generator:
          type: gamma  # rate is set by capacity search
        cancel_session_on_failure: false

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

      client:
        type: openai_chat_completions
        api_base: http://localhost:8000/v1
        model: meta-llama/Llama-3-8B-Instruct

      runtime:
        max_sessions: -1
        benchmark_timeout: 60

      evaluators:
        - type: performance
          target_channels: ["text"]
          slos:
            - name: "P99 TTFC"
              metric: ttfc
              percentile: 0.99
              value: 0.5
              type: constant
            - name: "P90 TBC"
              metric: tbc
              percentile: 0.90
              value: 0.05
              type: constant

Run the search:

.. code-block:: bash

    python -Xgil=0 -m veeksha.capacity_search --capacity-search-config-from-file capacity_search.veeksha.yml


Rate-based vs concurrency-based searches
----------------------------------------

**Rate-Based** (finding max sessions/second)

Use when you want to find the maximum arrival rate:

.. code-block:: yaml

    benchmark_config:
      traffic_scheduler:
        type: rate
        interval_generator:
          type: gamma  # or poisson

Capacity search sets the ``arrival_rate`` parameter.

**Concurrency-Based** (finding max concurrent sessions)

Use when you want to find maximum sustainable concurrency:

.. code-block:: yaml

    benchmark_config:
      traffic_scheduler:
        type: concurrent
        # target_concurrent_sessions and rampup_seconds are set by search

Capacity search sets ``target_concurrent_sessions``.


Configuration reference
-----------------------

.. code-block:: yaml

    output_dir: capacity_search_output    # Base output directory

    start_value: 5.0       # Initial probe value
    max_value: 100.0       # Maximum value to search
    expansion_factor: 2.0  # Probe multiplier (default: 2.0)
    max_iterations: 20     # Max total iterations
    precision: 2           # Decimal precision for rate searches

    benchmark_config:
      # Full benchmark configuration
      # See /config_reference/benchmark for all options

``start_value``
    Initial value for probing. Choose a value likely to pass.

``max_value``
    Upper bound for the search. Probing won't exceed this.

``expansion_factor``
    How aggressively to probe (2.0 = double each time).

``precision``
    For rate-based searches, how many decimal places to use.
    Set to 0 for integer concurrency searches.


Defining SLOs
-------------

SLOs determine pass/fail for each iteration. Define them in the evaluator:

.. code-block:: yaml

    evaluators:
      - type: performance
        slos:
          - name: "P99 TTFC under 500ms"
            metric: ttfc
            percentile: 0.99
            value: 0.5      # 500ms in seconds
            type: constant

          - name: "P99 TBC under 50ms"
            metric: tbc
            percentile: 0.99
            value: 0.05

          - name: "P95 E2E under 10s"
            metric: e2e
            percentile: 0.95
            value: 10.0

Available metrics:

- ``ttfc``: Time to first chunk/token
- ``tbc``: Time between chunks
- ``tpot``: Time per output token
- ``e2e``: End-to-end latency

An iteration **passes** only if **all** SLOs are met.


Output structure
----------------

.. code-block:: text

    capacity_search_output/
    └── 08:01:2026-17:25:35-b0fc8e1d/
        ├── config.yml                       # Search configuration
        ├── capacity_search_results.json     # Final results
        └── runs/                            # Individual benchmark runs
            ├── 08:01:2026-17:25:35-f24d1805/
            │   ├── config.yml
            │   ├── metrics/
            │   └── ...
            └── 08:01:2026-17:25:50-4ea72acb/
                └── ...

**capacity_search_results.json** contains the search outcome:

.. code-block:: json

    {
      "traffic_scheduler_type": "concurrent",
      "searched_knob": "traffic_scheduler.target_concurrent_sessions",
      "best_value": 20.0,
      "best_run_dir": "capacity_search_output/.../runs/...",
      "history": [
        {
          "value": 10.0,
          "all_slos_met": true,
          "run_dir": ".../runs/08:01:2026-17:25:35-f24d1805",
          "slo_results": { ... }
        },
        {
          "value": 20.0,
          "all_slos_met": true,
          "run_dir": ".../runs/08:01:2026-17:25:50-4ea72acb",
          "slo_results": { ... }
        }
      ]
    }


WandB integration
-----------------

Enable WandB to track all iterations and get a summary run:

.. code-block:: yaml

    benchmark_config:
      wandb:
        enabled: true
        project: veeksha
        group: cap-search-llama-8b

The summary run includes:

- Final capacity value
- All iterations plotted
- Comparison table
- "BEST_CONFIG" tag on the optimal run


Example: Production capacity planning
-------------------------------------

Find the maximum rate for a latency-sensitive deployment:

.. code-block:: yaml

    output_dir: capacity_search_prod

    start_value: 10.0
    max_value: 500.0
    expansion_factor: 2.0
    max_iterations: 15
    precision: 1

    benchmark_config:
      traffic_scheduler:
        type: rate
        interval_generator:
          type: poisson

      session_generator:
        type: trace
        trace_file: production_traffic.jsonl
        flavor:
          type: claude_code

      client:
        type: openai_chat_completions
        api_base: http://prod-server:8000/v1
        model: meta-llama/Llama-3-70B-Instruct

      runtime:
        benchmark_timeout: 120
        max_sessions: -1

      evaluators:
        - type: performance
          slos:
            - name: "P99 TTFT < 2s"
              metric: ttfc
              percentile: 0.99
              value: 2.0
            - name: "P50 TBC < 30ms"
              metric: tbc
              percentile: 0.50
              value: 0.03

This uses real production traces and strict SLOs for accurate capacity planning.
