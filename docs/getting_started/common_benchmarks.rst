Benchmark Types
===============

This page maps common benchmark shapes to their canonical Veeksha patterns so
you can quickly see how Veeksha fits the way you benchmark today. It is not
exhaustive; Veeksha is composable, so the same building blocks can be combined
into many other valid configurations.


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

The examples below show canonical starting points rather than the only possible
configurations.
More specialized workload patterns appear later on this page.


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

    uvx -p 3.14t veeksha benchmark --config rate_single_request.veeksha.yml


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

    uvx -p 3.14t veeksha benchmark --config concurrent_single_request.veeksha.yml


TTFC vs prompt length
---------------------

Use this when you want isolated prefill measurements.

.. code-block:: bash

    uvx -p 3.14t veeksha prefill \
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

    uvx -p 3.14t veeksha decode \
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

    uvx -p 3.14t veeksha stress \
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

    uvx -p 3.14t veeksha capacity-search --config capacity_search.veeksha.yml

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

    uvx -p 3.14t veeksha benchmark --config replay_request_log.veeksha.yml

Your trace file should contain ``input_length`` and ``output_length`` columns.
If you need a multi-turn conversation trace, a timed session trace, a
shared-prefix trace, or a RAG trace, see :doc:`/user_guide/trace_flavors` for a
flavor-by-flavor comparison and minimal trace examples.


Multi-turn conversations (synthetic)
------------------------------------

Use this when you want generated multi-turn chat rather than independent
requests:

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


.. _workload-recipes:

Advanced workload patterns
--------------------------

These examples cover more specialized benchmark types. Treat them as canonical
starting points, not as an exhaustive list of every supported configuration.

Unless noted otherwise, run them with:

.. code-block:: bash

    uvx -p 3.14t veeksha benchmark --config <file>.veeksha.yml

For trace-based workloads beyond simple request-log replay, including
conversation datasets, timed multi-turn traces, and shared-prefix traces, see
:doc:`/user_guide/trace_flavors`.


Agentic workloads (branching sessions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


See also
--------

- :doc:`quick_start` for a first end-to-end benchmark run
- :doc:`/user_guide/configuration` for the full config model
- :doc:`/user_guide/trace_flavors` for trace flavor details and input formats
- :doc:`/user_guide/microbenchmarks` for full prefill, decode, and stress details
- :doc:`/user_guide/capacity_search` for full capacity-search behavior

Accuracy (WER) — the QUALITY mode
---------------------------------

``mode: quality`` turns a benchmark into an accuracy measurement: exactly one
scored pass over a trace corpus, no wrapping, every session scored once. The
config is rejected at load time — before any server is touched — if it could
silently produce a number that is not the corpus WER (wrapped trace, or a
``max_sessions`` that scores a prefix or re-scores clips).

.. code-block:: yaml

   mode: quality
   session_generator:
     type: trace
     trace_file: traces/asr/aa_full/manifest_voxpopuli.jsonl
     wrap_mode: false          # required: wrapping re-scores clips
     flavor:
       type: audio
       audio_dir: ""
   traffic_scheduler:
     type: concurrent
     target_concurrent_sessions: 156   # pick a load your provider is stable at
     rampup_seconds: 0
   runtime:
     max_sessions: 628         # must equal the corpus size (distinct session_ids)
     benchmark_timeout: 1800
   client:
     type: stt                 # swap the client block to target another provider
     provider: vajra_openai_realtime
     api_base: http://127.0.0.1:9000
     model: voxtral-mini-realtime
     ws_realtime_pacing: true
     request_timeout: 300      # must exceed your LONGEST clip in seconds
   evaluators:
   - type: performance
     target_channels: [audio]
     stream_metrics: false

The summary reports corpus, duration-weighted, and per-dataset WER
(``asr_dataset_<name>_final_*``) when the manifest carries a ``dataset``
column, so a mixed corpus never hides one dataset behind another.

Provider-independence and two footguns:

* Only the ``client`` block is provider-specific — the mode's guarantees are
  enforced regardless of provider.
* ``request_timeout`` must exceed the longest clip: a timed-out session
  contributes an empty transcript and the run still exits 0.
* Streaming providers can transcribe differently when audio arrives faster
  than real time; keep pacing on (or your provider's equivalent) unless
  burst-delivery accuracy is itself the thing being measured.
