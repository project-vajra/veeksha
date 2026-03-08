Evaluation and Metrics
======================

Veeksha's evaluation system collects detailed metrics for every request and
computes aggregate statistics. This page explains the available metrics,
how they're computed, and how SLOs are evaluated.


Evaluator architecture
----------------------

Evaluators consume completed requests and produce metrics:

.. code-block:: python

    class BaseEvaluator(ABC):
        """Lifecycle:
        1. register_request() - when request is dispatched
        2. record_request_completed() - when response is received
        3. record_session_completed() - when session finishes
        4. finalize() - compute aggregate metrics
        5. save() - write to output directory
        """

Two evaluator types are available:

- **Performance** (``type: performance``): Latency, throughput, timing metrics
- **Accuracy** (``type: accuracy_lmeval``): Model evaluation using lm-eval-harness


Performance metrics
-------------------

The performance evaluator computes these key metrics:

**TTFC (Time to First Chunk/Token)**
    Time from request dispatch to receiving the first response token.
    Critical for user-perceived responsiveness.

    .. code-block:: text

        TTFC = first_token_timestamp - scheduler_dispatched_at

**TBC (Time Between Chunks/Tokens)**
    Average time between consecutive tokens. Affects streaming experience.

    .. code-block:: text

        TBC = (last_token_timestamp - first_token_timestamp) / (num_tokens - 1)

**TPOT (Time Per Output Token)**
    Average time per output token including TTFC. Overall generation speed.

    .. code-block:: text

        TPOT = (client_completed_at - client_picked_up_at) / num_output_tokens

**E2E (End-to-End) Latency**
    Total time from dispatch to completion.

    .. code-block:: text

        E2E = client_completed_at - scheduler_dispatched_at

**Throughput**
    Aggregate rates computed from all completed requests:

    - ``tpot_based_throughput``: Output tokens / total time
    - ``tbc_based_throughput``: Tokens/sec based on average TBC


Configuring evaluators
----------------------

Add evaluators to your benchmark configuration:

.. code-block:: yaml

    evaluators:
      - type: performance
        target_channels: ["text"]
        stream_metrics: true
        stream_metrics_interval: 5.0
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

``target_channels``
    List of channel modalities to evaluate. Usually ``["text"]``.

``stream_metrics``
    If ``true``, periodically logs metrics to console during the benchmark.

``stream_metrics_interval``
    Seconds between streaming metric updates.


SLO definitions
---------------

Service Level Objectives (SLOs) define acceptable performance thresholds:

.. code-block:: yaml

    slos:
      - name: "P99 TTFC under 500ms"
        metric: ttfc          # Metric to check
        percentile: 0.99      # Percentile level
        value: 0.5            # Threshold in seconds
        type: constant        # SLO type

Available metrics for SLOs:

- ``ttfc``: Time to first chunk
- ``tbc``: Time between chunks
- ``tpot``: Time per output token
- ``e2e``: End-to-end latency

SLO results are saved to ``metrics/slo_results.json``:

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
          "name": "P99 TTFC under 500ms",
          "lower_is_better": true
        }
      ]
    }


Output files
------------

The performance evaluator writes several files to ``metrics/``:

**Per-request data:**

``request_level_metrics.jsonl``
    JSON Lines file with detailed per-request data:

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


**Aggregate statistics:**

``summary_stats.json``
    High-level counts and rates:

    .. code-block:: json

        {
          "Number of Requests": 560,
          "Number of Completed Requests": 555,
          "Number of Errored Requests": 0,
          "Error Rate": 0.0,
          "Observed Session Dispatch Rate": 11.43
        }

``throughput_metrics.json``
    Throughput measurements:

    .. code-block:: json

        {
          "tpot_based_throughput": 76.99,
          "tbc_based_throughput": 11.32
        }

**Distribution files:**

For each metric (``ttfc``, ``tbc``, ``tpot``, ``e2e``, etc.):

- ``<metric>.csv``: Percentile values (p50, p90, p95, p99, min, max, mean)
- ``<metric>.png``: Distribution histogram


Prefill statistics
------------------

For understanding prefill latency scaling, the evaluator groups TTFC by prompt length:

``prefill_stats.json``:

.. code-block:: json

    {
      "metric": "ttfc",
      "group_by": "target_num_delta_prompt_tokens",
      "groups": {
        "128": {
          "count": 50,
          "mean": 0.034,
          "p99": 0.055
        },
        "256": {
          "count": 48,
          "mean": 0.041,
          "p99": 0.062
        }
      }
    }

This helps analyze how prefill time scales with prompt length.


Accuracy evaluation
-------------------

For model accuracy testing, use the lm-eval integration:

.. code-block:: yaml

    session_generator:
      type: lmeval
      tasks: ["hellaswag", "truthfulqa_gen"]
      num_fewshot: 0

    evaluators:
      - type: performance
        target_channels: ["text"]
      - type: accuracy_lmeval
        bootstrap_iters: 200

This runs lm-eval-harness tasks through Veeksha's load generation, allowing
simultaneous accuracy and performance measurement.

The accuracy evaluator outputs:

- Standard lm-eval metrics (accuracy, perplexity)
- Integration with standard lm-eval result formats


Streaming metrics
-----------------

During a benchmark, enable real-time metric output:

.. code-block:: yaml

    evaluators:
      - type: performance
        stream_metrics: true
        stream_metrics_interval: 5.0

This logs current statistics every 5 seconds:

.. code-block:: text

    [10.2s] Completed: 156 | TTFC p99: 45ms | TBC p99: 18ms | Throughput: 234 tok/s

Useful for monitoring long-running benchmarks without waiting for completion.


Health checks
-------------

After the benchmark, Veeksha runs health checks to validate correctness:

**Session Dispatch Rate Check**
    Verifies actual arrival rate matches configuration.

**Prompt Length Check**
    Verifies generated prompt lengths match targets.

**Output Length Check**
    Verifies output lengths match requested tokens (when server supports it).

**Lifecycle Timing Delays Check**
    Reports timing overhead at each pipeline stage.

**Intra-Session Request Arrival Check**
    Verifies request dependencies were respected.

Results are saved to ``health_check_results.txt``:

Health checks help identify configuration issues or system problems that
could invalidate benchmark results.
