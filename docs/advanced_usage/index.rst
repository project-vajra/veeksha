Advanced Usage
==============

This section covers advanced Veeksha features for power users and researchers
who need fine-grained control over benchmarking workflows.


In This Section
---------------

.. toctree::
   :maxdepth: 2

   capacity_search
   server_management
   sweeps
   microbenchmarks


Overview
--------

**Capacity Search**
    Automatically find the maximum sustainable rate or concurrency that meets
    your latency SLOs. Uses an adaptive probe-then-binary-search algorithm.

**Server Management**
    Let Veeksha launch and manage inference servers (vLLM, SGLang) automatically.
    Useful for CI pipelines and reproducible experiments.

**Configuration Sweeps**
    Run multiple benchmarks with different parameters using the ``!expand`` YAML
    tag. Creates Cartesian product of configurations.

**Microbenchmarks**
    Isolate and measure specific operations like prefill or decode throughput
    using specialized configurations.