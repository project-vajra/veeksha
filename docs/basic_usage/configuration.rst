Configuration System
====================

Veeksha uses a flexible polymorphic configuration system that supports YAML files,
CLI arguments, and programmatic access. This guide explains how the system works
and how to navigate it effectively.


Configuration Methods
---------------------

**YAML Files** (recommended)
    Create a ``.veeksha.yml`` file with your configuration:

    .. code-block:: yaml

        seed: 42
        client:
          type: openai_chat_completions
          api_base: http://localhost:8000/v1
          model: my-model
        traffic_scheduler:
          type: rate
          interval_generator:
            type: poisson
            arrival_rate: 10.0

**CLI Arguments**
    Override any option using dot notation:

    .. code-block:: bash

        python -Xgil=0 -m veeksha.benchmark \
            --client-api-base http://localhost:8000/v1 \
            --traffic-scheduler-interval-generator-arrival-rate 20.0

**Combined** (YAML + CLI)
    CLI arguments override YAML values:

    .. code-block:: bash

        # Base config from file, override arrival rate
        python -Xgil=0 -m veeksha.benchmark \
            --benchmark-config-from-file base.veeksha.yml \
            --traffic-scheduler-interval-generator-arrival-rate 30.0


Polymorphic Options
-------------------

Many options have a ``type`` field that selects a variant with its own options:

.. code-block:: yaml

    # Session generator can be: synthetic, trace, or lmeval
    session_generator:
      type: synthetic        # Selects synthetic variant
      session_graph:         # Options specific to synthetic
        type: linear
      channels:
        - type: text

    # Traffic scheduler can be: rate or concurrent
    traffic_scheduler:
      type: rate             # Selects rate variant
      interval_generator:    # Options specific to rate
        type: poisson
        arrival_rate: 10.0

Each ``type`` exposes different options. Use the config explorer to discover them.


Config Exploration Tools
------------------------

Veeksha includes CLI tools for exploring the configuration schema:

**Interactive Explorer**

.. code-block:: bash

    python -m veeksha.cli.config explore

To navigate the config tree interactively.

**Show Full Schema**

.. code-block:: bash

    # YAML format
    python -m veeksha.cli.config show --format yaml

    # JSON format
    python -m veeksha.cli.config show --format json

.. _configuration-export-json-schema:

**Export JSON Schema** (for YAML IDE autocompletion and linting)

.. code-block:: bash

    python -m veeksha.cli.config export-schema -o veeksha-schema.json

Configure your IDE to use this schema. In VSCode and forks:

.. code-block:: json

    // .vscode/settings.json
    {
        "yaml.schemas": {
            "./veeksha-schema.json": "*.veeksha.yml"
        },
        "yaml.customTags": [
            "!expand sequence"
        ]
    }

.. hint::
  The YAML IDE extension may be required for "yaml.schemas" to show up as a valid setting.

.. figure:: /_static/assets/yaml_help_text.png
   :alt: VSCode YAML integration example
   :align: center
   :width: 600px

   The VSCode YAML extension providing autocompletion and documentation on hover.




Common Configuration Sections
-----------------------------

**client** - API endpoint configuration

.. code-block:: yaml

    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct
      # api_key: optional, falls back to OPENAI_API_KEY env var
      request_timeout: 300
      max_tokens_param: max_completion_tokens
      min_tokens_param: min_tokens

**traffic_scheduler** - Traffic pattern

.. code-block:: yaml

    # Rate-based
    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 10.0
      cancel_session_on_failure: true

    # OR Concurrency-based
    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: 8
      rampup_seconds: 10

**session_generator** - Content generation

.. code-block:: yaml

    session_generator:
      type: synthetic
      session_graph:
        type: linear
        num_request_generator:
          type: uniform
          min: 1
          max: 5
        inherit_history: true
      channels:
        - type: text
          body_length_generator:
            type: uniform
            min: 100
            max: 500

**runtime** - Execution parameters

.. code-block:: yaml

    runtime:
      benchmark_timeout: 300      # Total benchmark duration
      max_sessions: 1000          # Maximum sessions (-1 = unlimited)
      post_timeout_grace_seconds: 10  # Wait for in-flight after timeout
      num_client_threads: 3       # Async HTTP client threads

**evaluators** - Metrics collection

.. code-block:: yaml

    evaluators:
      - type: performance
        target_channels: ["text"]
        stream_metrics: true
        slos:
          - name: "P99 TTFC"
            metric: ttfc
            percentile: 0.99
            value: 0.5
            type: constant


Environment Variables
---------------------

Veeksha automatically reads certain environment variables as fallbacks when
configuration values are not explicitly set:

``OPENAI_API_KEY``
    Used as the API key if ``client.api_key`` is not set in config.

``OPENAI_API_BASE``
    Used as the API base URL if ``client.api_base`` is not set in config.

This allows you to set credentials once in your environment:

.. code-block:: bash

    export OPENAI_API_KEY=your-api-key
    export OPENAI_API_BASE=http://localhost:8000/v1

Then omit them from your config file:

.. code-block:: yaml

    # No need to specify api_key or api_base
    client:
      type: openai_chat_completions
      model: meta-llama/Llama-3-8B-Instruct

This is especially useful for:

- Avoiding committing secrets to version control
- Sharing configs across environments with different servers


Stop Conditions
---------------

Benchmarks stop when either condition is met:

.. code-block:: yaml

    runtime:
      benchmark_timeout: 300    # Stop after 300 seconds
      max_sessions: 1000        # OR after 1000 sessions

Use ``-1`` for unlimited:

.. code-block:: yaml

    runtime:
      benchmark_timeout: -1     # Run indefinitely
      max_sessions: 500         # Stop only after 500 sessions

When a timeout hits, Veeksha will record all in-flight requests and keep dispatching sessions as usual. Then, it might exit after the ``post_timeout_grace_seconds``:

.. code-block:: yaml

    runtime:
      benchmark_timeout: 60
      post_timeout_grace_seconds: 10  # Wait 10s for in-flight requests
      # -1 = wait indefinitely for all in-flight
      # 0 = exit immediately (cancel in-flight)


Output Directory
----------------

Control where results are saved:

.. code-block:: yaml

    output_dir: benchmark_output

Results are saved to a timestamped subdirectory:

.. code-block:: text

    benchmark_output/
    └── 09:01:2026-10:30:00-a1b2c3d4/
        ├── config.yml
        ├── metrics/
        └── traces/

The subdirectory name includes:

- Date and time
- Short hash of the configuration (for uniqueness)


Trace Recording
---------------

Control what's recorded for debugging:

.. code-block:: yaml

    trace_recorder:
      enabled: true          # Write trace file
      include_content: false # Exclude prompt/response content (smaller files)

Set ``include_content: true`` to record full request content for debugging.


Validation
----------

Veeksha validates configurations at startup:

- Type checking for all fields
- Enum validation for ``type`` fields
- Required field checking
- Cross-field validation (e.g., ``min <= max``)

Invalid configurations produce clear error messages:

.. code-block:: text

    ConfigurationError: traffic_scheduler.interval_generator.arrival_rate
    must be positive, got -5.0


See Also
--------

- :doc:`/config_reference/benchmark` - Complete benchmark configuration reference
- :doc:`/config_reference/capacity_search` - Capacity search configuration reference
