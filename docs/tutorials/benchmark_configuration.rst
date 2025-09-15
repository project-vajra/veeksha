Benchmark Configuration
======================

This guide covers the key configuration options for running effective benchmarks with ``veeksha``. Rather than listing every parameter, this focuses on the options that significantly impact your benchmark results and testing strategy.

Core Benchmark Control
----------------------

Test Duration and Completion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The benchmark can end in two ways - by time limit or request count:

.. code-block:: shell

    python -m veeksha.benchmark \
        --timeout 1800 \
        --max-completed-requests 100

**timeout**: Sets maximum runtime in seconds. The benchmark will stop when this time is reached, even if ``max-completed-requests`` hasn't been met. Set to ``-1`` for no timeout (automatically set for LMEval tasks).

**max-completed-requests**: Stops the benchmark after completing this many requests, even if time remains. Useful for quick validation runs or resource-constrained testing.

Client Configuration
~~~~~~~~~~~~~~~~~~~

Control how many connections and concurrent requests to send:

.. code-block:: shell

    python -m veeksha.benchmark \
        --num-clients 4 \
        --num-concurrent-requests-per-client 8

**num-clients**: Number of separate client connections to the API. Increase for higher throughput testing.

**num-concurrent-requests-per-client**: Requests each client sends in parallel. Higher values test concurrency handling but may overwhelm unstable systems.

Total concurrent requests = ``num_clients × num_concurrent_requests_per_client``

Request Generation Strategies
----------------------------

Synthetic Request Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Synthetic generation creates artificial workloads using statistical distributions:

.. code-block:: shell

    python -m veeksha.benchmark \
        --request-generator-config-type "synthetic"

Length Generation Patterns
**************************

**Trace-based (Recommended)**: Uses real production traffic patterns

.. code-block:: shell

    python -m veeksha.benchmark \
        --synthetic-request-generator-config-length-generator-config-type "trace" \
        --synthetic-request-generator-config-trace-length-generator-config-trace-file "data/my_trace.csv" \
        --synthetic-request-generator-config-trace-length-generator-config-prefill-scale-factor 0.8 \
        --synthetic-request-generator-config-trace-length-generator-config-decode-scale-factor 1.2

- **prefill-scale-factor/decode-scale-factor**: Adjust trace data for different model capabilities. Scale down (0.5) for smaller models, up (2.0) for larger contexts.
- **exhaustion-policy**: What happens when trace data runs out - ``stop`` ends the test, ``wrap`` repeats the trace, ``error`` fails.

**Fixed**: Predictable lengths for controlled testing

.. code-block:: shell

    python -m veeksha.benchmark \
        --synthetic-request-generator-config-length-generator-config-type "fixed" \
        --synthetic-request-generator-config-fixed-length-generator-config-prefill-tokens 1024 \
        --synthetic-request-generator-config-fixed-length-generator-config-decode-tokens 256

**Uniform**: Random lengths within a range

.. code-block:: shell

    python -m veeksha.benchmark \
        --synthetic-request-generator-config-length-generator-config-type "uniform" \
        --synthetic-request-generator-config-uniform-length-generator-config-min-tokens 512 \
        --synthetic-request-generator-config-uniform-length-generator-config-prefill-to-decode-ratio 4.0

Interval Generation Patterns
****************************

**Poisson (Default)**: Natural request arrival with random intervals

.. code-block:: shell

    python -m veeksha.benchmark \
        --synthetic-request-generator-config-interval-generator-config-type "poisson" \
        --synthetic-request-generator-config-poisson-interval-generator-config-qps 2.5

**Gamma**: More realistic bursty traffic patterns

.. code-block:: shell

    python -m veeksha.benchmark \
        --synthetic-request-generator-config-interval-generator-config-type "gamma" \
        --synthetic-request-generator-config-gamma-interval-generator-config-qps 2.0 \
        --synthetic-request-generator-config-gamma-interval-generator-config-cv 1.0

- **cv (coefficient of variation)**: Controls traffic burstiness. Lower values (0.1) = steady traffic, higher values (2.0) = very bursty.

**Static**: Maximum sustained load testing

.. code-block:: shell

    python -m veeksha.benchmark \
        --synthetic-request-generator-config-interval-generator-config-type "static"

Sends all requests immediately with no delays. Use for testing maximum processing capability.

LMEval Request Generation
~~~~~~~~~~~~~~~~~~~~~~~~

For model evaluation benchmarks:

.. code-block:: shell

    python -m veeksha.benchmark \
        --request-generator-config-type "lmeval" \
        --lmeval-request-generator-config-tasks "hellaswag,arc_easy,winogrande" \
        --lmeval-request-generator-config-num-fewshot 5 \
        --lmeval-request-generator-config-limit 100

**tasks**: Which evaluation benchmarks to run. See LMEval documentation for available tasks.
**num-fewshot**: Number of example shots for consistency across evaluations.
**limit**: Maximum examples per task to control runtime and cost.

Performance Measurement
-----------------------

Deadline-based SLA Reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define acceptable performance thresholds:

.. code-block:: shell

    python -m veeksha.benchmark \
        --metrics-config-deadline-report-ttft-deadline 0.5 \
        --metrics-config-deadline-report-tbt-deadline 0.1 \
        --metrics-config-deadline-report-target-deadline-miss-rate 0.05

**ttft-deadline**: Time-to-first-token threshold in seconds. Critical for user experience - users notice delays over 0.3-0.5 seconds.

**tbt-deadline**: Time-between-tokens threshold in seconds. Essential for streaming - gaps over 0.1 seconds feel choppy.

**target-deadline-miss-rate**: Acceptable failure rate (0.05 = 95% of requests must meet deadlines). Used to calculate SLA compliance.

Experiment Tracking
~~~~~~~~~~~~~~~~~~

Integration with Weights & Biases:

.. code-block:: shell

    python -m veeksha.benchmark \
        --metrics-config-should-write-metrics-to-wandb \
        --metrics-config-wandb-project "llm-benchmarks" \
        --metrics-config-wandb-group "llama-3-8b" \
        --metrics-config-wandb-run-name "production-load-test"

System Integration
-----------------

API Compatibility
~~~~~~~~~~~~~~~~

**llm-api**: Automatically set based on task type, but can be overridden:

.. code-block:: shell

    # For text completion APIs
    python -m veeksha.benchmark \
        --client-config-llm-api "openai_completions" \
        --client-config-address-append-value "completions"
    
    # For chat completion APIs  
    python -m veeksha.benchmark \
        --client-config-llm-api "openai_chat" \
        --client-config-address-append-value "chat/completions"

Model Parameters
~~~~~~~~~~~~~~~

Pass sampling parameters that affect performance:

.. code-block:: shell

    python -m veeksha.benchmark \
        --client-config-additional-sampling-params '{"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}'

These parameters significantly impact resource usage - higher creativity settings (temperature/top_p) often require more computation.

Timeout Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    python -m veeksha.benchmark \
        --client-config-request-timeout 120

**request-timeout**: Per-request timeout in seconds. Should be higher than expected worst-case response time to avoid false failures, but low enough to detect actual hangs.

Common Configuration Patterns
-----------------------------

Quick Validation
~~~~~~~~~~~~~~~

Fast test to verify system is working:

.. code-block:: shell

    python -m veeksha.benchmark \
        --max-completed-requests 10 \
        --timeout 60 \
        --synthetic-request-generator-config-length-generator-config-type "fixed" \
        --synthetic-request-generator-config-fixed-length-generator-config-prefill-tokens 256 \
        --synthetic-request-generator-config-fixed-length-generator-config-decode-tokens 64

Production Load Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~

Realistic traffic patterns:

.. code-block:: shell

    python -m veeksha.benchmark \
        --timeout 3600 \
        --num-clients 8 \
        --num-concurrent-requests-per-client 4 \
        --synthetic-request-generator-config-poisson-interval-generator-config-qps 5.0 \
        --synthetic-request-generator-config-trace-length-generator-config-trace-file "prod_trace.csv"

Maximum Capacity Testing
~~~~~~~~~~~~~~~~~~~~~~~

Find breaking point:

.. code-block:: shell

    python -m veeksha.benchmark \
        --timeout 600 \
        --num-clients 16 \
        --num-concurrent-requests-per-client 8 \
        --synthetic-request-generator-config-interval-generator-config-type "static"