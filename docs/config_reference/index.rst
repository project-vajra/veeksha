Configuration Reference
=======================

This section provides a comprehensive reference for all configuration options in Veeksha.
Configuration can be provided via YAML files or CLI arguments.

.. tip::
   
   Use the interactive config explorer for an easier experience::
   
       python -m veeksha.cli.config explore

   Or generate a YAML schema template::
   
       python -m veeksha.cli.config show --format yaml


Quick Links
-----------

- :doc:`benchmark` - Configuration for standard benchmark runs
- :doc:`capacity_search` - Configuration for capacity search experiments


Understanding the Config System
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


IDE Autocompletion
------------------

Export a JSON schema for YAML autocompletion in your IDE::

    python -m veeksha.cli.config export-schema -o veeksha-schema.json

Then configure your IDE to use this schema for ``*.veeksha.yml`` files.

For VS Code, add to ``.vscode/settings.json``::

    {
        "yaml.schemas": {
            "./veeksha-schema.json": "*.veeksha.yml"
        }
    }


.. toctree::
   :maxdepth: 2
   :hidden:

   benchmark
   capacity_search
