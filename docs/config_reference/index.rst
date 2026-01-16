Configuration Reference
=======================

This section provides a comprehensive reference for all configuration options in Veeksha.
Configuration can be provided via YAML files or CLI arguments.

.. tip::
   
   Use the interactive config explorer for an easier experience::
   
       python -m veeksha.cli.config explore

   Or generate a YAML schema template::
   
       python -m veeksha.cli.config show --format yaml


Quick links
-----------

- :doc:`api_reference/BenchmarkConfig` - Configuration for standard benchmark runs
- :doc:`api_reference/CapacitySearchConfig` - Configuration for capacity search experiments
- :doc:`api_reference/index` - Full API reference for all config classes


Understanding the config system
-------------------------------

Veeksha uses a **polymorphic configuration system**. Many options have a ``type`` field
that determines which variant is used, each with its own set of options.

For example, the ``traffic_scheduler`` can be either ``rate`` or ``concurrent``::

    # Rate-based traffic
    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        rate: 10.0  # 10 requests per second

    # Concurrency-based traffic
    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: 8
      rampup_seconds: 10


IDE autocompletion
------------------

See :doc:`/basic_usage/configuration` for instructions on how to set up YAML autocompletion and validation in your IDE.


.. toctree::
   :maxdepth: 2
   :hidden:

   api_reference/index
