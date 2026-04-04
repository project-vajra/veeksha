Common Benchmarks
=================

If you are coming from another benchmarking framework, start here. Pick the benchmark
you want, copy the example, and run it. You can tune details later.


Pick the benchmark
------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - If you want to measure...
     - Use this in Veeksha
   * - Open-loop request-rate latency
     - ``benchmark`` with ``single_request`` sessions and ``rate`` traffic
   * - Closed-loop fixed-concurrency throughput
     - ``benchmark`` with ``single_request`` sessions and ``concurrent`` traffic
   * - TTFC vs prompt length
     - ``prefill``
   * - TBT/TPOT vs batch size
     - ``decode``
   * - Throughput vs latency curve
     - ``stress``
   * - Max sustainable rate or concurrency under SLOs
     - ``capacity-search``
   * - Replay a request log or conversation dataset
     - ``benchmark`` with ``trace`` sessions

For most request-level benchmarks, ``benchmark`` is the right command. Veeksha
models traffic as sessions, but ``single_request`` sessions make it behave like
a traditional request dispatcher.


Open-loop request-rate latency test
-----------------------------------

Use this when you would normally run a fixed-QPS or Poisson-arrival benchmark.

.. code-block:: yaml

    # rate_single_request.veeksha.yml
    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct

    session_generator:
      type: synthetic
      session_graph:
        type: single_request
      channels:
        - type: text
          body_length_generator:
            type: fixed
            value: 256
      output_spec:
        text:
          output_length_generator:
            type: fixed
            value: 128

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 5.0

    runtime:
      benchmark_timeout: 60
      max_sessions: -1

    evaluators:
      - type: performance
        target_channels: ["text"]

.. code-block:: bash

    uvx veeksha benchmark --config rate_single_request.veeksha.yml


Closed-loop fixed-concurrency throughput test
---------------------------------------------

Use this when you want to hold a target concurrency and push for throughput.

.. code-block:: yaml

    # concurrent_single_request.veeksha.yml
    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct

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
      target_concurrent_sessions: 16
      rampup_seconds: 10

    runtime:
      benchmark_timeout: 120
      max_sessions: -1

    evaluators:
      - type: performance
        target_channels: ["text"]

.. code-block:: bash

    uvx veeksha benchmark --config concurrent_single_request.veeksha.yml


TTFC vs prompt length
---------------------

Use this when you want isolated prefill measurements.

.. code-block:: bash

    uvx veeksha prefill \
        --api_base http://localhost:8000/v1 \
        --model meta-llama/Llama-3-8B-Instruct \
        --input_lengths 128 256 512 1024 2048 \
        --output_tokens 1 \
        --samples_per_length 10 \
        --output_dir microbench_output

This sweeps prompt length and keeps decode minimal so you can see how TTFC
scales with prefill work.


TBT/TPOT vs batch size
----------------------

Use this when you want isolated decode measurements.

.. code-block:: bash

    uvx veeksha decode \
        --api_base http://localhost:8000/v1 \
        --model meta-llama/Llama-3-8B-Instruct \
        --batch_sizes 1 2 4 8 16 \
        --input_lengths 128 512 \
        --samples_per_length 20 \
        --engine_chunk_size 512 \
        --output_dir microbench_output

This measures steady-state decode behavior as batching increases.


Throughput vs latency curve
---------------------------

Use this when you want the classic operating curve for one fixed request shape.

.. code-block:: bash

    uvx veeksha stress \
        --api_base http://localhost:8000/v1 \
        --model meta-llama/Llama-3-8B-Instruct \
        --input_length 512 \
        --output_length 256 \
        --mode.type manual \
        --mode.concurrency_levels 1 2 4 8 16 32 \
        --point_duration 120 \
        --warmup_duration 10 \
        --output_dir microbench_output

This gives you throughput, end-to-end latency, TTFC, and interactivity at each
concurrency level.


Max sustainable load under SLOs
-------------------------------

Use this when you want Veeksha to find the highest passing rate or concurrency
automatically.

.. code-block:: yaml

    # capacity_search.veeksha.yml
    output_dir: capacity_search_output
    start_value: 5.0
    max_value: 100.0
    expansion_factor: 2.0
    precision: 1

    benchmark_config:
      client:
        type: openai_chat_completions
        api_base: http://localhost:8000/v1
        model: meta-llama/Llama-3-8B-Instruct

      session_generator:
        type: synthetic
        session_graph:
          type: single_request
        channels:
          - type: text
            body_length_generator:
              type: fixed
              value: 256
        output_spec:
          text:
            output_length_generator:
              type: fixed
              value: 128

      traffic_scheduler:
        type: rate
        interval_generator:
          type: poisson

      runtime:
        benchmark_timeout: 60
        max_sessions: -1

      evaluators:
        - type: performance
          target_channels: ["text"]
          slos:
            - name: "P99 TTFC < 500ms"
              metric: ttfc
              percentile: 0.99
              value: 0.5
              type: constant
            - name: "P99 TBC < 50ms"
              metric: tbc
              percentile: 0.99
              value: 0.05
              type: constant

.. code-block:: bash

    uvx veeksha capacity-search --config capacity_search.veeksha.yml

For concurrency instead of rate, change ``benchmark_config.traffic_scheduler.type``
to ``concurrent`` and set ``precision: 0``.


Replay a request log
--------------------

Use this when you already have a CSV or JSONL file with input and output lengths.

.. code-block:: yaml

    # replay_request_log.veeksha.yml
    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct

    session_generator:
      type: trace
      trace_file: requests.csv
      wrap_mode: true
      flavor:
        type: request_log

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 10.0

    runtime:
      benchmark_timeout: 120
      max_sessions: -1

    evaluators:
      - type: performance
        target_channels: ["text"]

.. code-block:: bash

    uvx veeksha benchmark --config replay_request_log.veeksha.yml

Your trace file should contain ``input_length`` and ``output_length`` columns.
If you need a multi-turn conversation trace, a timed session trace, a
shared-prefix trace, or a RAG trace, see :doc:`/user_guide/trace_flavors` for a
flavor-by-flavor comparison and minimal trace examples.


When you need chat, not just requests
-------------------------------------

Many competing tools stop at independent requests. To benchmark multi-turn chat
in Veeksha, keep using ``benchmark`` and change the session graph:

.. code-block:: yaml

    session_generator:
      type: synthetic
      session_graph:
        type: linear
        inherit_history: true
        num_request_generator:
          type: uniform
          min: 2
          max: 4

Everything else stays the same. This turns a request benchmark into a real
conversation benchmark.


See also
--------

- :doc:`quick_start` for a first end-to-end benchmark run
- :doc:`/user_guide/configuration` for the full config model
- :doc:`/user_guide/microbenchmarks` for full prefill, decode, and stress details
- :doc:`/user_guide/capacity_search` for full capacity-search behavior
