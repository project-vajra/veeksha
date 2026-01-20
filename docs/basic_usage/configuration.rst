Configuration System
====================

Veeksha uses a flexible polymorphic configuration system that supports YAML files,
CLI arguments, and programmatic access. This guide explains how the system works
and how to navigate it effectively.


Configuration methods
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
    Override any option using dash notation:

    .. code-block:: bash

        python -Xgil=0 -m veeksha.benchmark \
            --openai-chat-completions-client-api-base http://localhost:8000/v1 \
            --rate-traffic-scheduler-poisson-interval-generator-arrival-rate 20.0

Note how argument names contain their type.      

**Combined** (YAML + CLI)
    CLI arguments override YAML values:

    .. code-block:: bash

        # Base config from file, override arrival rate
        python -Xgil=0 -m veeksha.benchmark \
            --benchmark-config-from-file base.veeksha.yml \
            --rate-traffic-scheduler-poisson-interval-generator-arrival-rate 30.0


Polymorphic options
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

Each ``type`` exposes different options. Use the config explorer (next section) to discover them or take a look at the :doc:`/config_reference/index`.


Config exploration tools
------------------------

On top of the full API reference (:doc:`/config_reference/index`), Veeksha includes CLI tools for exploring the configuration schema:

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

**Export JSON schema** (for YAML IDE autocompletion and linting)

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


Common configuration sections
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
      output_spec:
        text:
          output_length_generator:
            type: uniform
            min: 50
            max: 200

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


Environment variables
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

Veeksha also reads ``HF_TOKEN`` from the environment in order to access gated models.

Stop conditions
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

When a timeout hits, Veeksha will record all in-flight requests and keep dispatching sessions as usual. 
Then, it will exit after ``post_timeout_grace_seconds`` have passed, only if the session limit is not reached before that. 

.. code-block:: yaml

    runtime:
      benchmark_timeout: 60
      post_timeout_grace_seconds: 10  # Wait 10s for in-flight requests
      # -1 = wait indefinitely for all in-flight
      # 0 = exit immediately (cancel in-flight)


Output directory
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


Trace recording
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


.. _configuration-splitting:

Splitting configuration across files
-------------------------------------

For better organization and reusability, you can split your configuration across
multiple YAML files. This is useful when you want to:

- Reuse client configuration across different benchmarks
- Keep environment-specific settings (e.g., API endpoints) separate
- Share traffic patterns across experiments

**Example: Separate client and content configs**

Create ``client.yml`` with just client settings:

.. code-block:: yaml

    # client.yml
    type: openai_chat_completions
    api_base: http://localhost:8000/v1
    model: meta-llama/Llama-3-8B-Instruct

Create ``synthetic_content.yml`` with benchmark settings:

.. code-block:: yaml

    # synthetic_content.yml
    seed: 42
    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 5.0
    session_generator:
      type: synthetic
      channels:
        - type: text
          body_length_generator:
            type: uniform
            min: 50
            max: 200
    runtime:
      benchmark_timeout: 60

Run the benchmark combining both:

.. code-block:: bash

    python -Xgil=0 -m veeksha.benchmark \
        --benchmark-config-from-file synthetic_content.yml \
        --client-from-file client.yml

**Available from-file arguments**

The ``--*-from-file`` argument pattern follows the config class hierarchy. Currently
supported for root-level dataclass fields:

For ``veeksha.benchmark``:

- ``--benchmark-config-from-file`` - Load ``BenchmarkConfig`` from file
- ``--client-from-file`` - Load ``client`` from file
- ``--session-generator-from-file`` - Load ``session_generator`` from file
- ``--traffic-scheduler-from-file`` - Load ``traffic_scheduler`` from file
- ``--runtime-from-file`` - Load ``runtime`` from file

For ``veeksha.capacity_search``:

- ``--capacity-search-config-from-file`` - Load ``CapacitySearchConfig`` from file
- ``--benchmark-config-client-from-file`` - Load ``benchmark_config.client`` from file

.. note::
    The argument names follow the field names in the config classes. You can
    discover them via the config explorer (``python -m veeksha.cli.config explore``)
    or by checking the config class definitions in ``veeksha/config/``.

**CLI overrides still work**

You can override any value from the files using CLI arguments:

.. code-block:: bash

    python -Xgil=0 -m veeksha.benchmark \
        --benchmark-config-from-file synthetic_content.yml \
        --client-from-file client.yml \
        --openai-chat-completions-client-model llama-70b  # Override model

**Collision behavior**

Each configuration section can only be defined once. If the same section appears
in multiple files, an error is raised:

.. code-block:: text

    # ERROR: client defined in both files
    python -Xgil=0 -m veeksha.benchmark \
        --benchmark-config-from-file full_config.yml \  # Contains client section
        --client-from-file client.yml                   # Also defines client

To avoid collisions, ensure each config file defines **disjoint** sections of the
configuration tree.


See also
--------

- :doc:`/config_reference/api_reference/BenchmarkConfig` - Complete benchmark configuration reference
- :doc:`/config_reference/api_reference/CapacitySearchConfig` - Capacity search configuration reference
