Session Generators
==================

Session generators enable realistic conversation-based benchmarking by grouping requests into sessions (conversations) based on prefix similarity and controlling session dispatch rates independently from conversation pacing. This is particularly useful for evaluating prefix cache performance and simulating realistic user interaction patterns.

The session generator takes traces with hash IDs and timestamps, clusters them into sessions using prefix matching, and then dispatches entire sessions at controlled intervals while preserving the original timing within each conversation.

Session Generation Overview
---------------------------

When you have trace data with hash IDs (representing request prefixes), the session generator:

1. **Groups requests into sessions** based on prefix similarity using hash ID matching
2. **Controls session dispatch rate** - how often new conversations start
3. **Preserves conversation pacing** - maintains original timing between requests within each session
4. **Simulates prefix cache behavior** - calculates cache hit rates based on prefix overlap

This approach allows you to modulate overall system load by controlling how often users start conversations, without artificially slowing down individual conversation flows.

Session Configuration
--------------------

Basic Session Generation
~~~~~~~~~~~~~~~~~~~~~~~~

To enable session-based request generation, use a trace with hash IDs and configure the session generator:

.. code-block:: shell

    python -m veeksha.benchmark \
        # ... other arguments
        --request-generator-config-type "trace" \
        --trace-request-generator-config-trace-file "path/to/hash_trace.jsonl" \
        --trace-request-generator-config-use-session-generator \
        --trace-request-generator-config-session-generator-config-minimum-prefix-match 0.7 \
        --trace-request-generator-config-session-generator-config-min-session-size 2 \
        --trace-request-generator-config-session-generator-config-max-session-size 20 \
        --seed 42

In this example, requests are grouped into sessions when they share at least 70% prefix similarity, with sessions containing 2-20 requests each.

Session Clustering Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**minimum-prefix-match** (0.8): Controls how similar requests must be to group into the same session. Higher values (0.9) create tighter sessions with very similar prefixes, while lower values (0.5) allow more diverse conversations.

**min-session-size** (1): Filters out sessions with too few requests. Set higher (3-5) to focus on substantial conversations and remove single-request sessions.

**max-session-size** (10): Limits session length to prevent overly long conversations from skewing results. Typical values are 10-50 depending on your use case.

**max-request-interval** (600.0): Maximum time gap between requests within a session (seconds). Conversations with longer pauses are split into separate sessions.

Session Dispatch Control
~~~~~~~~~~~~~~~~~~~~~~~

Control how often new sessions (conversations) start using interval generators:

**Poisson Session Dispatch** (Default):

.. code-block:: shell

    python -m veeksha.benchmark \
        # ... other arguments
        --trace-request-generator-config-session-generator-config-session-interval-generator-config-type "poisson" \
        --trace-request-generator-config-session-generator-config-poisson-session-interval-generator-config-qps 0.5 \
        --seed 42

This starts new conversations following a Poisson distribution with an average of 0.5 new sessions per second.

**Gamma Session Dispatch**:

.. code-block:: shell

    python -m veeksha.benchmark \
        # ... other arguments
        --trace-request-generator-config-session-generator-config-session-interval-generator-config-type "gamma" \
        --trace-request-generator-config-session-generator-config-gamma-session-interval-generator-config-qps 1.0 \
        --trace-request-generator-config-session-generator-config-gamma-session-interval-generator-config-cv 2.0 \
        --seed 42

Creates bursty session arrivals with higher variance (CV=2.0), simulating periods of high and low user activity.

**Static Session Dispatch**:

.. code-block:: shell

    python -m veeksha.benchmark \
        # ... other arguments
        --trace-request-generator-config-session-generator-config-session-interval-generator-config-type "static"

Dispatches all sessions immediately for maximum load testing.

Trace File Format
----------------

Session-based generation requires trace files with hash IDs. Your trace file should be in JSONL format with these fields:

.. code-block:: json

    {"request_id": "req_001", "timestamp": 1234567890.123, "hash_ids": [100, 101, 102, 103]}
    {"request_id": "req_002", "timestamp": 1234567892.456, "hash_ids": [100, 101, 104]}
    {"request_id": "req_003", "timestamp": 1234567895.789, "hash_ids": [200, 201]}

**hash_ids**: List of integers representing the prefix structure of each request. Requests with overlapping prefixes will be grouped into sessions.

**timestamp**: Request timestamp in seconds (can be floating point).

Saving Generated Sessions
-------------------------

Save the processed session data as a new trace file for reuse:

.. code-block:: shell

    python -m veeksha.benchmark \
        # ... other arguments
        --trace-request-generator-config-session-generator-config-save-as-trace-file \
        --trace-request-generator-config-session-generator-config-trace-file-save-dir "./processed_traces" \
        --trace-request-generator-config-session-generator-config-trace-file-name "my_session_trace"

The saved trace will include session metadata and can be reused for consistent benchmarking.

Use Cases
---------

Prefix Cache Evaluation
~~~~~~~~~~~~~~~~~~~~~~

Evaluate how well your system's prefix cache performs with realistic conversation patterns:

.. code-block:: shell

    python -m veeksha.benchmark \
        --trace-request-generator-config-session-generator-config-minimum-prefix-match 0.8 \
        --trace-request-generator-config-session-generator-config-poisson-session-interval-generator-config-qps 2.0 \
        --trace-request-generator-config-session-generator-config-max-session-size 15

This simulates 2 new conversations per second with high prefix similarity (80%) and realistic conversation lengths.

Load Scaling Simulation
~~~~~~~~~~~~~~~~~~~~~~

Test how your system performs under different user loads while maintaining realistic conversation dynamics:

.. code-block:: shell

    # Low load - few conversations starting
    python -m veeksha.benchmark \
        --trace-request-generator-config-session-generator-config-poisson-session-interval-generator-config-qps 0.2
    
    # High load - many conversations starting  
    python -m veeksha.benchmark \
        --trace-request-generator-config-session-generator-config-poisson-session-interval-generator-config-qps 5.0

The individual conversation pacing remains realistic, but you control how many concurrent conversations are active.

Bursty Traffic Simulation
~~~~~~~~~~~~~~~~~~~~~~~~

Simulate realistic user behavior with periods of high and low activity:

.. code-block:: shell

    python -m veeksha.benchmark \
        --trace-request-generator-config-session-generator-config-session-interval-generator-config-type "gamma" \
        --trace-request-generator-config-session-generator-config-gamma-session-interval-generator-config-cv 3.0

High coefficient of variation creates realistic traffic bursts where many users start conversations simultaneously, followed by quieter periods.