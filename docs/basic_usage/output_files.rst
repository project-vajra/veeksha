Output Files
============

Every benchmark run creates a timestamped output directory containing
configuration, metrics, traces, and verification results. This guide
explains each file and how to use them.


Output Directory Structure
--------------------------

.. code-block:: text

    benchmark_output/
    └── 09:01:2026-10:30:00-a1b2c3d4/
        ├── config.yml                    # Resolved configuration
        ├── health_check_results.txt      # Benchmark verification
        ├── wandb_run.json                # WandB run info (if enabled)
        ├── metrics/
        │   ├── request_level_metrics.jsonl
        │   ├── summary_stats.json
        │   ├── throughput_metrics.json
        │   ├── slo_results.json
        │   ├── prefill_stats.json
        │   ├── ttfc.csv / ttfc.png
        │   ├── tbc.csv / tbc.png
        │   ├── tpot.csv / tpot.png
        │   ├── end_to_end_latency.csv / .png
        │   └── ... (other metric files)
        ├── traces/
        │   └── trace.jsonl
        └── wandb/                        # WandB local files (if enabled)

The directory name format is: ``DD:MM:YYYY-HH:MM:SS-<config_hash>``


Configuration File
------------------

``config.yml``
    The fully resolved configuration used for the benchmark:

    .. code-block:: yaml

        output_dir: benchmark_output/09:01:2026-10:30:00-a1b2c3d4
        seed: 42
        session_generator:
          type: synthetic
          session_graph:
            type: linear
            inherit_history: true
            num_request_generator:
              type: uniform
              min: 2
              max: 6
        # ... all resolved values

    This file enables exact reproduction of the benchmark.


Metrics Directory
-----------------

**request_level_metrics.jsonl**
    JSON Lines file with one record per completed request:

    .. code-block:: json

        {
          "request_id": 75, 
          "session_id": 8, 
          "session_total_requests": 8, 
          "scheduler_ready_at": 0.53709, 
          "scheduler_dispatched_at": 0.53709, 
          "client_picked_up_at": 0.53723, 
          "client_completed_at": 0.60328, 
          "result_processed_at": 0.60346, 
          "num_delta_prompt_tokens": 6, 
          "num_total_prompt_tokens": 6, 
          "target_num_delta_prompt_tokens": 6, 
          "num_output_tokens": 7, 
          "num_requested_output_tokens": 7, 
          "num_total_tokens": 13, 
          "is_stream": true, 
          "tpot": 0.00688, 
          "ttfc": 0.0243, 
          "end_to_end_latency": 0.06559, 
          "normalized_end_to_end_latency": 0.00937, 
          "output_throughput": 106.72167, 
          "tbc": [0.00814, 0.0067, 0.00687, 0.00611, 0.00658, 0.00689]
        }

    Key fields:

    - **Timing**: Timestamps at each lifecycle stage
    - **Latencies**: ``ttfc``, ``tbc``, ``tpot`` in seconds
    - **Tokens**: Prompt and output token counts
    - **Status**: ``success``, ``error_code``

**summary_stats.json**
    Aggregate statistics:

    .. code-block:: json

        {
          "Number of Requests": 560,
          "Number of Completed Requests": 555,
          "Number of Errored Requests": 0,
          "Number of Cancelled Requests": 5,
          "Error Rate": 0.0,
          "Cancellation Rate": 0.009,
          "Number of Sessions Seen": 110,
          "Successful Sessions": 100,
          "Errored Sessions": 0,
          "Cancelled Sessions": 0.0,
          "Incomplete Sessions": 10,
          "Observed Session Dispatch Rate": 11.43
        }

**throughput_metrics.json**
    Throughput measurements:

    .. code-block:: json

        {
          "tpot_based_throughput": 76.99,
          "tbc_based_throughput": 11.32
        }

    - ``tpot_based_throughput``: Total output tokens / total time
    - ``tbc_based_throughput``: Throughput based on average TBC

**slo_results.json**
    SLO evaluation results (if SLOs were configured):

    .. code-block:: json

        {
          "all_slos_met": true,
          "results": [
            {
              "met": true,
              "slo_metric_key": "ttfc_p99",
              "observed_value": 0.055,
              "threshold": 0.5,
              "percentile": 0.99,
              "metric": "ttfc",
              "name": "P99 TTFC",
              "lower_is_better": true
            }
          ]
        }

    Used by capacity search to determine pass/fail.

**prefill_stats.json**
    TTFC grouped by prompt length for prefill analysis:

    .. code-block:: json

        {
          "metric": "ttfc",
          "group_by": "target_num_delta_prompt_tokens",
          "groups": {
            "128": {"count": 50, "mean": 0.034, "p99": 0.055},
            "256": {"count": 48, "mean": 0.041, "p99": 0.062}
          }
        }


Metric Distribution Files
-------------------------

For each metric, a CSV and PNG file are generated:

**ttfc.csv** - Time to First Chunk percentiles:

.. code-block:: text

    ,cdf,Time to First Chunk
    0,0.0,0.0169...
    1,0.01,0.0182...
    2,0.02,0.0187...
    3,0.03,0.0191...
    4,0.04,0.0195...
    5,0.05,0.0203...
    6,0.06,0.0209...
    7,0.07,0.0213...
    8,0.08,0.021...
    9,0.09,0.0218...
    ...
    100,1.0,0.176...

**ttfc.png** - Distribution histogram showing TTFC values across all requests.

*(Similar files exist for tbc, tpot, e2e, session_duration, etc.)*


Traces Directory
----------------

**trace.jsonl**
    Dispatched request traces (if ``trace_recorder.enabled: true``):

    .. code-block:: json

        {
          "request_id": 42,
          "session_id": 7,
          "session_size": 21
          "dispatched_at": 0.49085,
          "session_context": {
            "node_id": 0,
            "wait_after_ready": 0,
            "parent_nodes": [],
            "history_parent": null
          }
        }

    With ``include_content: true``, also includes:

    - Full prompt text/tokens
    - Target token lengths
    - History from parent requests


Health Check Results
--------------------

``health_check_results.txt``
    Verification that the benchmark ran correctly:

    .. code-block:: text

        ============================================================
        SESSION DISPATCH RATE CHECK
        ============================================================
        Result: PASSED

        Rate Statistics:
          Total Sessions                 110
          Expected Rate                  10.0000 sessions/sec
          Actual Rate                    11.4026 sessions/sec
          Error                          14.03%
          Threshold                      15.0%

        ============================================================
        PROMPT LENGTH CHECK
        ============================================================
        Result: PASSED

        Statistics:
          Total Requests Checked         555
          Exact Matches                  555 (100.0%)
          Violations (> +/-15)           0 (0.0%)

        ============================================================
        LIFECYCLE TIMING DELAYS CHECK
        ============================================================
        Result: PASSED

        Dispatch-to-Pickup Delay:
          Mean                           0.0006s
          P99                            0.0030s

    Checks included:

    - **Session Dispatch Rate**: Arrival rate accuracy
    - **Intra-Session Request Arrival**: Dependency timing
    - **Prompt Length**: Target vs actual prompt tokens
    - **Output Length**: Target vs actual output tokens
    - **Lifecycle Timing Delays**: Pipeline overhead


WandB Files
-----------

If WandB is enabled:

**wandb_run.json**
    Basic run identifiers:

    .. code-block:: json

        {
          "run_id": "abc123xyz",
          "run_name": "09:01:2026-10:30:00-a1b2c3d4",
          "run_url": "https://wandb.ai/entity/project/runs/abc123xyz"
        }

**wandb/**
    Local WandB sync directory containing logs and artifacts.


Analyzing Results Programmatically
----------------------------------

Load metrics in Python:

.. code-block:: python

    import json
    import pandas as pd
    from pathlib import Path

    output_dir = Path("benchmark_output/09:01:2026-10:30:00-a1b2c3d4")

    # Load per-request metrics
    metrics = pd.read_json(
        output_dir / "metrics/request_level_metrics.jsonl",
        lines=True
    )

    # Compute statistics
    print(f"Median TTFC: {metrics['ttfc'].median() * 1000:.1f} ms")
    print(f"P99 TTFC: {metrics['ttfc'].quantile(0.99) * 1000:.1f} ms")
    print(f"Mean TBC: {metrics['tbc'].mean() * 1000:.1f} ms")

    # Load summary
    with open(output_dir / "metrics/summary_stats.json") as f:
        summary = json.load(f)
    print(f"Completed: {summary['Number of Completed Requests']}")

    # Check SLO results
    with open(output_dir / "metrics/slo_results.json") as f:
        slos = json.load(f)
    print(f"All SLOs met: {slos['all_slos_met']}")


See Also
--------

- :doc:`/understanding_veeksha/evaluation` - How metrics are computed
