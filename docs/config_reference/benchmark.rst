Benchmark Configuration
=======================

Configuration reference for ``veeksha.benchmark`` runs.

Example Configuration
---------------------

.. code-block:: yaml

    benchmark_config:
      output_dir: <str>  # default: 'benchmark_output'
      seed: <int>  # default: 42
      session_generator:
        type: <synthetic | lmeval | trace>  # default: <class 'veeksha.config.generator.session.SyntheticSessionGeneratorConfig'>
        # synthetic:
        #   session_graph:
        #     type: <linear>  # default: <class 'veeksha.config.generator.session_graph.LinearSessionGraphGeneratorConfig'>
        #     inherit_history: <bool>  # default: True
        #     # linear:
        #     #   inherit_history: <bool>  # default: True
        #     #   num_request_generator:
        #     #     type: <fixed | fixed_stair | uniform | zipf>  # default: <class 'veeksha.config.generator.length.UniformLengthGeneratorConfig'>
        #     #     # fixed:
        #     #     #   value: <int>  # default: 8
        #     #     # fixed_stair:
        #     #     #   values: <List[int]>  # default: <callable>
        #     #     #   repeat_each: <int>  # default: 1
        #     #     #   wrap: <bool>  # default: True
        #     #     # uniform:
        #     #     #   min: <int>  # default: 6
        #     #     #   max: <int>  # default: 12
        #     #     # zipf:
        #     #     #   theta: <float>  # default: 0.6
        #     #     #   scramble: <bool>  # default: False
        #     #     #   min: <int>  # default: 6
        #     #     #   max: <int>  # default: 12
        #     #   request_wait_generator:
        #     #     type: <gamma | poisson | fixed>  # default: <class 'veeksha.config.generator.interval.PoissonIntervalGeneratorConfig'>
        #     #     # gamma:
    # ... (see full schema below)

Full Reference
--------------

``output_dir``
~~~~~~~~~~~~~~

Base directory for all benchmark outputs (traces, metrics, logs)

**Type:** ``str``

**Default:** ``"benchmark_output"``

``seed``
~~~~~~~~

Seed for the random number generator.

**Type:** ``int``

**Default:** ``42``

``session_generator``
~~~~~~~~~~~~~~~~~~~~~

The session generator configuration for the benchmark. Available: synthetic, lmeval, trace.

**Type:** ``polymorphic (polymorphic)``

**Default:** ``<SyntheticSessionGeneratorConfig>``

**Available types:** ``synthetic``, ``lmeval``, ``trace``

**Type-specific options:**

When ``type: synthetic``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``session_graph``
     - ``polymorphic*``
     - <LinearSessionGraphGeneratorConfig>
     - The generator for the session graphs. Available: linear.
   * - ``channels``
     - ``list[List[polymorphic]]*``
     - []
     - The modality channels for the content of each request. Available: text, image, audio, video.


``session_generator.session_graph``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The generator for the session graphs. Available: linear.

**Type:** ``polymorphic (polymorphic)``

**Default:** ``<LinearSessionGraphGeneratorConfig>``

**Available types:** ``linear``

**Common options** (available for all types):

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``inherit_history``
     - ``bool``
     - True
     - Whether subsequent requests can inherit history from previous ones.


**Type-specific options:**

When ``type: linear``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``inherit_history``
     - ``bool``
     - True
     - Whether subsequent requests can inherit history from previous ones.
   * - ``num_request_generator``
     - ``polymorphic*``
     - <UniformLengthGeneratorConfig>
     - The generator for the number of requests. Available: zipf, uniform, fixed, fixed_stair.
   * - ``request_wait_generator``
     - ``polymorphic*``
     - <PoissonIntervalGeneratorConfig>
     - The generator for the wait time between requests. Available: poisson, gamma, fixed.


``session_generator.session_graph.num_request_generator``
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The generator for the number of requests. Available: zipf, uniform, fixed, fixed_stair.

**Type:** ``polymorphic (polymorphic)``

**Default:** ``<UniformLengthGeneratorConfig>``

**Available types:** ``fixed``, ``fixed_stair``, ``uniform``, ``zipf``

**Type-specific options:**

When ``type: fixed``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``value``
     - ``int``
     - 8
     - Value to generate.


When ``type: fixed_stair``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``values``
     - ``list[List[int]]``
     - [8, 16, 32, 64]
     - Ordered list of step values to emit.
   * - ``repeat_each``
     - ``int``
     - 1
     - Number of consecutive emissions per step value before advancing.
   * - ``wrap``
     - ``bool``
     - True
     - If True, cycle back to the first value after the last. If False, keep returning the last value.


When ``type: uniform``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``min``
     - ``int``
     - 6
     - Minimum value to generate.
   * - ``max``
     - ``int``
     - 12
     - Maximum value to generate.


When ``type: zipf``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``theta``
     - ``float``
     - 0.6
     - Theta parameter for the Zipf distribution.
   * - ``scramble``
     - ``bool``
     - False
     - Whether to scramble the Zipf distribution.
   * - ``min``
     - ``int``
     - 6
     - Minimum value to generate.
   * - ``max``
     - ``int``
     - 12
     - Maximum value to generate.


``session_generator.session_graph.request_wait_generator``
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The generator for the wait time between requests. Available: poisson, gamma, fixed.

**Type:** ``polymorphic (polymorphic)``

**Default:** ``<PoissonIntervalGeneratorConfig>``

**Available types:** ``gamma``, ``poisson``, ``fixed``

**Type-specific options:**

When ``type: gamma``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``arrival_rate``
     - ``float``
     - 1.0
     - Arrival rate for the Gamma distribution.
   * - ``cv``
     - ``float``
     - 0.5
     - Coefficient of variation for the Gamma distribution.


When ``type: poisson``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``arrival_rate``
     - ``float``
     - 1.0
     - Arrival rate for the Poisson distribution.


When ``type: fixed``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``interval``
     - ``float``
     - 1.0
     - Fixed interval for the fixed distribution.


``session_generator.channels``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The modality channels for the content of each request. Available: text, image, audio, video.

**Type:** ``list[List[polymorphic]] (polymorphic)``

**Default:** ``[]``

**Available types:** ``text``, ``image``, ``audio``, ``video``

**Type-specific options:**

When ``type: text``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``body_length_generator``
     - ``polymorphic*``
     - <UniformLengthGeneratorConfig>
     - The generator for the body length. Available: zipf, uniform, fixed, fixed_stair.
   * - ``output_length_generator``
     - ``polymorphic*``
     - <UniformLengthGeneratorConfig>
     - The generator for the output length. Available: zipf, uniform, fixed, fixed_stair.
   * - ``shared_prefix_ratio``
     - ``float``
     - 0.0
     - Fraction of prompt tokens to use as shared prefix for root requests (0.0-1.0)
   * - ``shared_prefix_probability``
     - ``float``
     - 1.0
     - Probability that a root request uses shared prefix (0.0-1.0)


``session_generator.channels.body_length_generator``
""""""""""""""""""""""""""""""""""""""""""""""""""""

The generator for the body length. Available: zipf, uniform, fixed, fixed_stair.

**Type:** ``polymorphic (polymorphic)``

**Default:** ``<UniformLengthGeneratorConfig>``

**Available types:** ``fixed``, ``fixed_stair``, ``uniform``, ``zipf``

**Type-specific options:**

When ``type: fixed``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``value``
     - ``int``
     - 8
     - Value to generate.


When ``type: fixed_stair``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``values``
     - ``list[List[int]]``
     - [8, 16, 32, 64]
     - Ordered list of step values to emit.
   * - ``repeat_each``
     - ``int``
     - 1
     - Number of consecutive emissions per step value before advancing.
   * - ``wrap``
     - ``bool``
     - True
     - If True, cycle back to the first value after the last. If False, keep returning the last value.


When ``type: uniform``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``min``
     - ``int``
     - 6
     - Minimum value to generate.
   * - ``max``
     - ``int``
     - 12
     - Maximum value to generate.


When ``type: zipf``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``theta``
     - ``float``
     - 0.6
     - Theta parameter for the Zipf distribution.
   * - ``scramble``
     - ``bool``
     - False
     - Whether to scramble the Zipf distribution.
   * - ``min``
     - ``int``
     - 6
     - Minimum value to generate.
   * - ``max``
     - ``int``
     - 12
     - Maximum value to generate.


``session_generator.channels.output_length_generator``
""""""""""""""""""""""""""""""""""""""""""""""""""""""

The generator for the output length. Available: zipf, uniform, fixed, fixed_stair.

**Type:** ``polymorphic (polymorphic)``

**Default:** ``<UniformLengthGeneratorConfig>``

**Available types:** ``fixed``, ``fixed_stair``, ``uniform``, ``zipf``

**Type-specific options:**

When ``type: fixed``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``value``
     - ``int``
     - 8
     - Value to generate.


When ``type: fixed_stair``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``values``
     - ``list[List[int]]``
     - [8, 16, 32, 64]
     - Ordered list of step values to emit.
   * - ``repeat_each``
     - ``int``
     - 1
     - Number of consecutive emissions per step value before advancing.
   * - ``wrap``
     - ``bool``
     - True
     - If True, cycle back to the first value after the last. If False, keep returning the last value.


When ``type: uniform``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``min``
     - ``int``
     - 6
     - Minimum value to generate.
   * - ``max``
     - ``int``
     - 12
     - Maximum value to generate.


When ``type: zipf``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``theta``
     - ``float``
     - 0.6
     - Theta parameter for the Zipf distribution.
   * - ``scramble``
     - ``bool``
     - False
     - Whether to scramble the Zipf distribution.
   * - ``min``
     - ``int``
     - 6
     - Minimum value to generate.
   * - ``max``
     - ``int``
     - 12
     - Maximum value to generate.


When ``type: image``:

*No additional options.*

When ``type: audio``:

*No additional options.*

When ``type: video``:

*No additional options.*

When ``type: lmeval``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``tasks``
     - ``list[List[str]]``
     - ['hellaswag']
     - The lm-eval tasks to evaluate the model on.
   * - ``num_fewshot``
     - ``int``
     - 1
     - The number of fewshot examples to use for the tasks.


When ``type: trace``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``trace_file``
     - ``str``
     - ""
     - Path to the JSONL trace file
   * - ``wrap_mode``
     - ``bool``
     - True
     - Whether to wrap/loop over the trace indefinitely
   * - ``flavor``
     - ``polymorphic*``
     - <ClaudeCodeTraceFlavorConfig>
     - Trace flavor configuration. Available: claude_code, mooncake_conv, rag.


``session_generator.flavor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trace flavor configuration. Available: claude_code, mooncake_conv, rag.

**Type:** ``polymorphic (polymorphic)``

**Default:** ``<ClaudeCodeTraceFlavorConfig>``

**Available types:** ``claude_code``, ``mooncake_conv``, ``rag``

**Type-specific options:**

When ``type: claude_code``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``corpus_file``
     - ``str``
     - "traces/corpus.txt"
     - Path to corpus file for prompt padding
   * - ``page_size``
     - ``int``
     - 16
     - Number of unique tokens per session prefix


When ``type: mooncake_conv``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``corpus_file``
     - ``str``
     - "traces/corpus.txt"
     - Path to corpus file for prompt padding
   * - ``block_size``
     - ``int``
     - 512
     - Number of tokens per hash id block. Only used for hash ids of first-in-session requests.


When ``type: rag``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``num_documents``
     - ``int``
     - 10
     - Number of top documents to include for warmup


``traffic_scheduler``
~~~~~~~~~~~~~~~~~~~~~

The traffic scheduler configuration for the benchmark. Available: rate, concurrent.

**Type:** ``polymorphic (polymorphic)``

**Default:** ``<RateTrafficConfig>``

**Available types:** ``rate``, ``concurrent``

**Common options** (available for all types):

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``cancel_session_on_failure``
     - ``bool``
     - True
     - Whether to cancel the session on failure of any request.


**Type-specific options:**

When ``type: rate``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``cancel_session_on_failure``
     - ``bool``
     - True
     - Whether to cancel the session on failure of any request.
   * - ``interval_generator``
     - ``polymorphic*``
     - <PoissonIntervalGeneratorConfig>
     - Interval generator for the traffic (sessions per second). Available: poisson, gamma, fixed.


``traffic_scheduler.interval_generator``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interval generator for the traffic (sessions per second). Available: poisson, gamma, fixed.

**Type:** ``polymorphic (polymorphic)``

**Default:** ``<PoissonIntervalGeneratorConfig>``

**Available types:** ``gamma``, ``poisson``, ``fixed``

**Type-specific options:**

When ``type: gamma``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``arrival_rate``
     - ``float``
     - 1.0
     - Arrival rate for the Gamma distribution.
   * - ``cv``
     - ``float``
     - 0.5
     - Coefficient of variation for the Gamma distribution.


When ``type: poisson``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``arrival_rate``
     - ``float``
     - 1.0
     - Arrival rate for the Poisson distribution.


When ``type: fixed``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``interval``
     - ``float``
     - 1.0
     - Fixed interval for the fixed distribution.


When ``type: concurrent``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``cancel_session_on_failure``
     - ``bool``
     - True
     - Whether to cancel the session on failure of any request.
   * - ``target_concurrent_sessions``
     - ``int``
     - 3
     - Target number of concurrent sessions to maintain.
   * - ``rampup_seconds``
     - ``int``
     - 10
     - Number of seconds to ramp up the traffic. i.e. 'Take 10 seconds to ramp up to the target concurrent sessions.'


``evaluators``
~~~~~~~~~~~~~~

List of evaluators to run. Available: performance, accuracy_lmeval.

**Type:** ``list[List[polymorphic]] (polymorphic)``

**Default:** ``[PerformanceEvaluatorConfig(target_channels=[<ChannelModality.TEXT: 1>], slos=[ConstantSloConfig(percentile=0.99, name='P99 TTFC', metric='ttfc', value=0.5), ConstantSloConfig(percentile=0.9, name='P90 TBC', metric='tbc', value=0.05)], stream_metrics=True, stream_metrics_interval=5.0, text_channel=TextChannelPerformanceConfig(decode_window_enabled=False, decode_window_config=None), image_channel=ImageChannelPerformanceConfig(), audio_channel=None, video_channel=None)]``

**Available types:** ``performance``, ``accuracy_lmeval``

**Common options** (available for all types):

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``target_channels``
     - ``list``
     - ['text']
     - List of ChannelModality values to evaluate.
   * - ``slos``
     - ``list[List[polymorphic]]*``
     - [ConstantSloConfig(percentile=0.99, name='P99 TTFC', metric='ttfc', value=0.5), ConstantSloConfig(percentile=0.9, name='P90 TBC', metric='tbc', value=0.05)]
     - List of SLO definitions to evaluate against request-level metrics. Available: constant.
   * - ``stream_metrics``
     - ``bool``
     - True
     - Enable real-time metric streaming
   * - ``stream_metrics_interval``
     - ``float``
     - 5.0
     - Interval for streaming metrics in seconds


**Type-specific options:**

When ``type: performance``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``target_channels``
     - ``list``
     - ['text']
     - List of ChannelModality values to evaluate.
   * - ``slos``
     - ``list[List[polymorphic]]*``
     - [ConstantSloConfig(percentile=0.99, name='P99 TTFC', metric='ttfc', value=0.5), ConstantSloConfig(percentile=0.9, name='P90 TBC', metric='tbc', value=0.05)]
     - List of SLO definitions to evaluate against request-level metrics. Available: constant.
   * - ``stream_metrics``
     - ``bool``
     - True
     - Enable real-time metric streaming
   * - ``stream_metrics_interval``
     - ``float``
     - 5.0
     - Interval for streaming metrics in seconds
   * - ``text_channel``
     - ``polymorphic*``
     - <TextChannelPerformanceConfig>
     - Text channel performance configuration
   * - ``image_channel``
     - ``polymorphic*``
     - <ImageChannelPerformanceConfig>
     - Image channel performance configuration
   * - ``audio_channel``
     - ``polymorphic*``
     - None
     - Audio channel performance configuration
   * - ``video_channel``
     - ``polymorphic*``
     - None
     - Video channel performance configuration


``evaluators.slos``
^^^^^^^^^^^^^^^^^^^

List of SLO definitions to evaluate against request-level metrics. Available: constant.

**Type:** ``list[List[polymorphic]] (polymorphic)``

**Default:** ``[ConstantSloConfig(percentile=0.99, name='P99 TTFC', metric='ttfc', value=0.5), ConstantSloConfig(percentile=0.9, name='P90 TBC', metric='tbc', value=0.05)]``

**Available types:** ``constant``

**Common options** (available for all types):

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``percentile``
     - ``float``
     - 0.99
     - Percentile at which to evaluate the SLO (0.0-1.0)
   * - ``name``
     - ``str``
     - None
     - Human-readable name for this SLO


**Type-specific options:**

When ``type: constant``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``percentile``
     - ``float``
     - 0.99
     - Percentile at which to evaluate the SLO (0.0-1.0)
   * - ``name``
     - ``str``
     - None
     - Human-readable name for this SLO
   * - ``metric``
     - ``str``
     - "ttfc"
     - The metric key this SLO applies to. Available: e2e, tbc, tpot, ttfc.
   * - ``value``
     - ``float``
     - -1.0
     - The constant value for the SLO. If a percentage, from 0 to 1. If a time, in seconds.


``evaluators.text_channel``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Text channel performance configuration

**Type:** ``polymorphic (polymorphic)``

**Default:** ``<TextChannelPerformanceConfig>``

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``decode_window_enabled``
     - ``bool``
     - False
     - Enable decode window analysis
   * - ``decode_window_config``
     - ``DecodeWindowConfig``
     - None
     - Decode window configuration (required if enabled)


``evaluators.text_channel.decode_window_config``
""""""""""""""""""""""""""""""""""""""""""""""""

Decode window configuration (required if enabled)

**Type:** ``DecodeWindowConfig``

**Default:** ``None``

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``min_active_requests``
     - ``int | str``
     - 1
     - Minimum number of simultaneously generating (decoding) requests required for a time interval to be considered inside the decode window. Use 'max_observed' to auto-detect the peak concurrent decoding count.
   * - ``selection_strategy``
     - ``str``
     - "longest"
     - Which window(s) to analyze when multiple windows exist. Supported: 'longest' (single longest), 'first' (single first), 'all' (aggregate all qualifying windows).
   * - ``anchor_to_client_pickup``
     - ``bool``
     - True
     - If True, anchor per-request token times to client_picked_up_at when available; otherwise use scheduler_dispatched_at.
   * - ``require_streaming``
     - ``bool``
     - True
     - If True, only streaming requests contribute to decode window analysis.


When ``type: accuracy_lmeval``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``target_channels``
     - ``list``
     - ['text']
     - List of ChannelModality values to evaluate.
   * - ``slos``
     - ``list[List[polymorphic]]*``
     - [ConstantSloConfig(percentile=0.99, name='P99 TTFC', metric='ttfc', value=0.5), ConstantSloConfig(percentile=0.9, name='P90 TBC', metric='tbc', value=0.05)]
     - List of SLO definitions to evaluate against request-level metrics. Available: constant.
   * - ``stream_metrics``
     - ``bool``
     - True
     - Enable real-time metric streaming
   * - ``stream_metrics_interval``
     - ``float``
     - 5.0
     - Interval for streaming metrics in seconds
   * - ``bootstrap_iters``
     - ``int``
     - 100000
     - Bootstrap iterations for confidence intervals


``evaluators.slos``
^^^^^^^^^^^^^^^^^^^

List of SLO definitions to evaluate against request-level metrics. Available: constant.

**Type:** ``list[List[polymorphic]] (polymorphic)``

**Default:** ``[ConstantSloConfig(percentile=0.99, name='P99 TTFC', metric='ttfc', value=0.5), ConstantSloConfig(percentile=0.9, name='P90 TBC', metric='tbc', value=0.05)]``

**Available types:** ``constant``

**Common options** (available for all types):

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``percentile``
     - ``float``
     - 0.99
     - Percentile at which to evaluate the SLO (0.0-1.0)
   * - ``name``
     - ``str``
     - None
     - Human-readable name for this SLO


**Type-specific options:**

When ``type: constant``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``percentile``
     - ``float``
     - 0.99
     - Percentile at which to evaluate the SLO (0.0-1.0)
   * - ``name``
     - ``str``
     - None
     - Human-readable name for this SLO
   * - ``metric``
     - ``str``
     - "ttfc"
     - The metric key this SLO applies to. Available: e2e, tbc, tpot, ttfc.
   * - ``value``
     - ``float``
     - -1.0
     - The constant value for the SLO. If a percentage, from 0 to 1. If a time, in seconds.


``client``
~~~~~~~~~~

The client configuration for the benchmark. Available: openai_chat_completions, openai_completions, openai_router.

**Type:** ``polymorphic (polymorphic)``

**Default:** ``<OpenAIChatCompletionsClientConfig>``

**Available types:** ``openai_chat_completions``, ``openai_completions``, ``openai_router``

**Common options** (available for all types):

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``api_base``
     - ``str``
     - None
     - API base URL. Defaults to OPENAI_API_BASE env var.
   * - ``api_key``
     - ``str``
     - None
     - API key. Defaults to OPENAI_API_KEY env var.
   * - ``model``
     - ``str``
     - "meta-llama/Meta-Llama-3-8B-Instruct"
     - The model to use for this load test.
   * - ``address_append_value``
     - ``str``
     - "chat/completions"
     - The address append value for the LLM API.
   * - ``request_timeout``
     - ``int``
     - 300
     - The timeout for each request to the LLM API (in seconds).
   * - ``additional_sampling_params``
     - ``str``
     - "{}"
     - Additional sampling params to send with each request to the LLM API.


**Type-specific options:**

When ``type: openai_chat_completions``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``api_base``
     - ``str``
     - None
     - API base URL. Defaults to OPENAI_API_BASE env var.
   * - ``api_key``
     - ``str``
     - None
     - API key. Defaults to OPENAI_API_KEY env var.
   * - ``model``
     - ``str``
     - "meta-llama/Meta-Llama-3-8B-Instruct"
     - The model to use for this load test.
   * - ``address_append_value``
     - ``str``
     - "chat/completions"
     - The address append value for the LLM API.
   * - ``request_timeout``
     - ``int``
     - 300
     - The timeout for each request to the LLM API (in seconds).
   * - ``additional_sampling_params``
     - ``str``
     - "{}"
     - Additional sampling params to send with each request to the LLM API.
   * - ``max_tokens_param``
     - ``str``
     - "max_completion_tokens"
     - Server parameter name for maximum tokens.
   * - ``min_tokens_param``
     - ``str``
     - "min_tokens"
     - Server parameter name for minimum tokens. If your server supports min tokens control via a parameter, specify its name here.
   * - ``use_min_tokens_prompt_fallback``
     - ``bool``
     - False
     - If True, appends instructions to the prompt to generate at least N tokens (e.g. 'Generate at least 20 tokens'). Useful if the server does not support a min tokens parameter. Only available on synthetic content generation.


When ``type: openai_completions``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``api_base``
     - ``str``
     - None
     - API base URL. Defaults to OPENAI_API_BASE env var.
   * - ``api_key``
     - ``str``
     - None
     - API key. Defaults to OPENAI_API_KEY env var.
   * - ``model``
     - ``str``
     - "meta-llama/Meta-Llama-3-8B-Instruct"
     - The model to use for this load test.
   * - ``address_append_value``
     - ``str``
     - "completions"
     - The address append value for the LLM API.
   * - ``request_timeout``
     - ``int``
     - 300
     - The timeout for each request to the LLM API (in seconds).
   * - ``additional_sampling_params``
     - ``str``
     - "{}"
     - Additional sampling params to send with each request to the LLM API.
   * - ``max_tokens_param``
     - ``str``
     - "max_tokens"
     - Server parameter name for maximum tokens.
   * - ``min_tokens_param``
     - ``str``
     - "min_tokens"
     - Server parameter name for minimum tokens. If your server supports min tokens control via a parameter, specify its name here.
   * - ``use_min_tokens_prompt_fallback``
     - ``bool``
     - False
     - If True, appends instructions to the prompt to generate at least N tokens (e.g. 'Generate at least 20 tokens'). Useful if the server does not support a min tokens parameter. Only available on synthetic content generation.


When ``type: openai_router``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``api_base``
     - ``str``
     - None
     - API base URL. Defaults to OPENAI_API_BASE env var.
   * - ``api_key``
     - ``str``
     - None
     - API key. Defaults to OPENAI_API_KEY env var.
   * - ``model``
     - ``str``
     - "meta-llama/Meta-Llama-3-8B-Instruct"
     - The model to use for this load test.
   * - ``address_append_value``
     - ``str``
     - "chat/completions"
     - The address append value for the LLM API.
   * - ``request_timeout``
     - ``int``
     - 300
     - The timeout for each request to the LLM API (in seconds).
   * - ``additional_sampling_params``
     - ``str``
     - "{}"
     - Additional sampling params to send with each request to the LLM API.
   * - ``max_tokens_param``
     - ``str``
     - "max_completion_tokens"
     - Server parameter name for maximum tokens.
   * - ``min_tokens_param``
     - ``str``
     - "min_tokens"
     - Server parameter name for minimum tokens. If your server supports min tokens control via a parameter, specify its name here.
   * - ``use_min_tokens_prompt_fallback``
     - ``bool``
     - False
     - If True, appends instructions to the prompt to generate at least N tokens (e.g. 'Generate at least 20 tokens'). Useful if the server does not support a min tokens parameter. Only available on synthetic content generation.
   * - ``completions_max_tokens_param``
     - ``str``
     - "max_tokens"
     - Server parameter name for maximum tokens on /completions endpoint. Defaults to 'max_tokens'. The /chat/completions endpoint uses max_tokens_param instead.


``runtime``
~~~~~~~~~~~

The runtime configuration for the benchmark.

**Type:** ``RuntimeConfig``

**Default:** ``<RuntimeConfig>``

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``max_sessions``
     - ``int``
     - 25
     - Maximum number of sessions to generate. -1 for unlimited.
   * - ``benchmark_timeout``
     - ``int``
     - 300
     - Benchmark timeout in seconds.
   * - ``post_timeout_grace_seconds``
     - ``int``
     - -1
     - Grace period for in-flight requests after timeout. -1 waits for all, 0 exits immediately.
   * - ``num_dispatcher_threads``
     - ``int``
     - 2
     - Number of threads for dispatching requests to workers.
   * - ``num_completion_threads``
     - ``int``
     - 2
     - Number of threads for processing completed requests.
   * - ``num_client_threads``
     - ``int``
     - 3
     - Number of async worker threads for making concurrent requests.


``trace_recorder``
~~~~~~~~~~~~~~~~~~

Trace recorder configuration. Records dispatched requests (unlike the evaluator, which records them after completion).

**Type:** ``TraceRecorderConfig``

**Default:** ``<TraceRecorderConfig>``

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``enabled``
     - ``bool``
     - True
     - Enable recording of dispatched requests
   * - ``include_content``
     - ``bool``
     - False
     - Include content of the request (channel blobs, history) in trace


``server``
~~~~~~~~~~

Server configuration for managed servers. If set, client.model, client.api_key and client.api_base will be overwritten.

**Type:** ``polymorphic (polymorphic)``

**Default:** ``None``

**Available types:** ``vllm``, ``vajra``, ``sglang``

**Common options** (available for all types):

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``env_path``
     - ``str``
     - None
     - Path to a Python environment directory (virtualenv/conda).
   * - ``model``
     - ``str``
     - "meta-llama/Meta-Llama-3-8B-Instruct"
     - Model name or path.
   * - ``host``
     - ``str``
     - "localhost"
     - Host address for the server
   * - ``port``
     - ``int``
     - 8000
     - Port number for the server
   * - ``api_key``
     - ``str``
     - "token-abc123"
     - API key for server authentication
   * - ``gpu_ids``
     - ``list[List[int]]``
     - None
     - List of GPU IDs to use (None means auto-assign)
   * - ``startup_timeout``
     - ``int``
     - 300
     - Timeout in seconds for server startup
   * - ``health_check_interval``
     - ``float``
     - 2.0
     - Interval in seconds between health checks
   * - ``require_contiguous_gpus``
     - ``bool``
     - True
     - Require contiguous GPU allocation (e.g., GPUs 0,1,2 instead of 0,2,5)
   * - ``tensor_parallel_size``
     - ``int``
     - 1
     - Number of GPUs for tensor parallelism
   * - ``dtype``
     - ``str``
     - "auto"
     - Data type for model weights (auto, float16, bfloat16, etc.)
   * - ``max_model_len``
     - ``int``
     - None
     - Maximum model context length
   * - ``additional_args``
     - ``str``
     - "{}"
     - Additional engine-specific arguments as JSON string, dict, or None.


**Type-specific options:**

When ``type: vllm``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``env_path``
     - ``str``
     - None
     - Path to a Python environment directory (virtualenv/conda).
   * - ``model``
     - ``str``
     - "meta-llama/Meta-Llama-3-8B-Instruct"
     - Model name or path.
   * - ``host``
     - ``str``
     - "localhost"
     - Host address for the server
   * - ``port``
     - ``int``
     - 8000
     - Port number for the server
   * - ``api_key``
     - ``str``
     - "token-abc123"
     - API key for server authentication
   * - ``gpu_ids``
     - ``list[List[int]]``
     - None
     - List of GPU IDs to use (None means auto-assign)
   * - ``startup_timeout``
     - ``int``
     - 300
     - Timeout in seconds for server startup
   * - ``health_check_interval``
     - ``float``
     - 2.0
     - Interval in seconds between health checks
   * - ``require_contiguous_gpus``
     - ``bool``
     - True
     - Require contiguous GPU allocation (e.g., GPUs 0,1,2 instead of 0,2,5)
   * - ``tensor_parallel_size``
     - ``int``
     - 1
     - Number of GPUs for tensor parallelism
   * - ``dtype``
     - ``str``
     - "auto"
     - Data type for model weights (auto, float16, bfloat16, etc.)
   * - ``max_model_len``
     - ``int``
     - None
     - Maximum model context length
   * - ``additional_args``
     - ``str``
     - "{}"
     - Additional engine-specific arguments as JSON string, dict, or None.


When ``type: vajra``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``env_path``
     - ``str``
     - None
     - Path to a Python environment directory (virtualenv/conda).
   * - ``model``
     - ``str``
     - "meta-llama/Meta-Llama-3-8B-Instruct"
     - Model name or path.
   * - ``host``
     - ``str``
     - "localhost"
     - Host address for the server
   * - ``port``
     - ``int``
     - 8000
     - Port number for the server
   * - ``api_key``
     - ``str``
     - "token-abc123"
     - API key for server authentication
   * - ``gpu_ids``
     - ``list[List[int]]``
     - None
     - List of GPU IDs to use (None means auto-assign)
   * - ``startup_timeout``
     - ``int``
     - 300
     - Timeout in seconds for server startup
   * - ``health_check_interval``
     - ``float``
     - 2.0
     - Interval in seconds between health checks
   * - ``require_contiguous_gpus``
     - ``bool``
     - True
     - Require contiguous GPU allocation (e.g., GPUs 0,1,2 instead of 0,2,5)
   * - ``tensor_parallel_size``
     - ``int``
     - 1
     - Number of GPUs for tensor parallelism
   * - ``dtype``
     - ``str``
     - "auto"
     - Data type for model weights (auto, float16, bfloat16, etc.)
   * - ``max_model_len``
     - ``int``
     - None
     - Maximum model context length
   * - ``additional_args``
     - ``str``
     - "{}"
     - Additional engine-specific arguments as JSON string, dict, or None.


When ``type: sglang``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``env_path``
     - ``str``
     - None
     - Path to a Python environment directory (virtualenv/conda).
   * - ``model``
     - ``str``
     - "meta-llama/Meta-Llama-3-8B-Instruct"
     - Model name or path.
   * - ``host``
     - ``str``
     - "localhost"
     - Host address for the server
   * - ``port``
     - ``int``
     - 8000
     - Port number for the server
   * - ``api_key``
     - ``str``
     - "token-abc123"
     - API key for server authentication
   * - ``gpu_ids``
     - ``list[List[int]]``
     - None
     - List of GPU IDs to use (None means auto-assign)
   * - ``startup_timeout``
     - ``int``
     - 300
     - Timeout in seconds for server startup
   * - ``health_check_interval``
     - ``float``
     - 2.0
     - Interval in seconds between health checks
   * - ``require_contiguous_gpus``
     - ``bool``
     - True
     - Require contiguous GPU allocation (e.g., GPUs 0,1,2 instead of 0,2,5)
   * - ``tensor_parallel_size``
     - ``int``
     - 1
     - Number of GPUs for tensor parallelism
   * - ``dtype``
     - ``str``
     - "auto"
     - Data type for model weights (auto, float16, bfloat16, etc.)
   * - ``max_model_len``
     - ``int``
     - None
     - Maximum model context length
   * - ``additional_args``
     - ``str``
     - "{}"
     - Additional engine-specific arguments as JSON string, dict, or None.


``wandb``
~~~~~~~~~

Weights & Biases logging configuration.

**Type:** ``WandbConfig``

**Default:** ``<WandbConfig>``

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Option
     - Type
     - Default
     - Description
   * - ``enabled``
     - ``bool``
     - False
     - Enable Weights & Biases logging.
   * - ``project``
     - ``str``
     - None
     - WandB project name (or set WANDB_PROJECT).
   * - ``entity``
     - ``str``
     - None
     - WandB entity (team/user). Optional.
   * - ``group``
     - ``str``
     - None
     - WandB group name (for sweeps/capacity-search).
   * - ``run_name``
     - ``str``
     - None
     - WandB run name. Defaults to the resolved output dir name.
   * - ``tags``
     - ``list[List[str]]``
     - []
     - List of WandB tags to attach to the run.
   * - ``notes``
     - ``str``
     - None
     - Optional WandB notes for this run.
   * - ``mode``
     - ``str``
     - None
     - Optional wandb mode override: 'online', 'offline', or 'disabled'.
   * - ``log_artifacts``
     - ``bool``
     - True
     - Upload selected output files as a wandb Artifact.

