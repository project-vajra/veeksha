Configuration Guide
===================

This guide covers how to configure ``veeksha`` benchmarks, including basic usage, advanced features, and troubleshooting.

.. contents:: Table of Contents
   :local:
   :depth: 2

Basic Configuration
-------------------

``veeksha`` can be configured in two ways:

1. **Command-line arguments**: Pass configuration directly via CLI flags
2. **Configuration files**: Use YAML files for more complex configurations

Using Configuration Files
^^^^^^^^^^^^^^^^^^^^^^^^^^

Configuration files use YAML format and can be specified with the ``--benchmark-config-from-file`` flag:

.. code-block:: bash

    python -m veeksha.benchmark --benchmark-config-from-file config.yml

Example configuration file:

.. code-block:: yaml

    timeout: 10000
    max_completed_requests: 100
    seed: 42

    client_config:
      endpoint:
        api_url: http://localhost:30000/v1
        api_key: token-abc123
      model: meta-llama/Meta-Llama-3-8B-Instruct
      num_clients: 15
      num_concurrent_requests_per_client: 10
      llm_api: openai_chat

    request_generator_config:
      type: synthetic
      interval_generator_config:
        type: poisson
      length_generator_config:
        type: uniform

Configuration Explosion
-----------------------

``veeksha`` automatically creates multiple benchmark configurations from a single config file when list values are provided for certain fields. This is called **config explosion**.

How It Works
^^^^^^^^^^^^

When you provide a list of values for a field that is **not** typed as a list, ``veeksha`` will automatically create separate configurations for each value.

**Example**: Multiple endpoints

.. code-block:: yaml

    client_config:
      endpoint:
        - name: endpoint-A
          api_url: http://localhost:8000/v1
          api_key: token-abc123
        - name: endpoint-B
          api_url: http://localhost:8002/v1
          api_key: token-def456

This creates **2 separate benchmark configurations**, one for each endpoint.

Cartesian Product
^^^^^^^^^^^^^^^^^

When multiple fields have list values, ``veeksha`` creates all possible combinations (Cartesian product):

.. code-block:: yaml

    client_config:
      endpoint:
        - name: endpoint-A
          api_url: http://localhost:8000/v1
        - name: endpoint-B
          api_url: http://localhost:8002/v1
      num_clients:
        - 10
        - 20

This creates **4 configurations**:
- endpoint A + 10 clients
- endpoint A + 20 clients
- endpoint B + 10 clients
- endpoint B + 20 clients

Advanced Features
-----------------

The ``!explode`` Tag
^^^^^^^^^^^^^^^^^^^^

By default, fields that are typed as ``List[...]`` in the code will **not** be exploded. The ``!explode`` YAML tag allows you to force explosion even for list-typed fields.

**Use Case**: Running separate benchmarks for different tasks

.. code-block:: yaml

    request_generator_config:
      type: lmeval
      # tasks is List[str] typed - normally would not explode
      # !explode forces creation of 3 separate configs
      tasks: !explode
        - hellaswag
        - winogrande
        - arc_easy
      num_fewshot: 1
      limit: 10

**Result**: Creates 3 separate benchmark configurations, one for each task.

When to Use ``!explode``
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``!explode`` tag when:

1. You want to run separate benchmarks for each item in a list-typed field
2. You need to create multiple configurations from a field that normally wouldn't explode
3. You want to make config explosion explicit and self-documenting

Combining ``!explode`` with Other Lists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``!explode`` tag works with the normal config explosion to create Cartesian products:

.. code-block:: yaml

    client_config:
      endpoint:  # Implicit explosion (not List-typed)
        - http://localhost:8000/v1
        - http://localhost:8002/v1

    request_generator_config:
      type: lmeval
      tasks: !explode  # Explicit explosion (IS List-typed)
        - hellaswag
        - winogrande

**Result**: Creates **4 configurations** (2 endpoints × 2 tasks)

Examples
--------

See example configuration files in the ``veeksha/benchmark_config_files/`` directory:

- ``example_synthetic.yml`` - Basic synthetic workload
- ``example_trace.yml`` - Trace-based workload
