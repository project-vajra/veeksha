Microbenchmarks
===============

``veeksha`` provides specialized microbenchmarking capabilities to measure detailed performance characteristics of LLM inference systems. The microbenchmarking system focuses on profiling two critical phases of LLM inference:

- **Prefill latency** (time-to-first-token)
- **Decode throughput** (time-between-tokens)

These measurements are performed across different prompt lengths and batch sizes to enable comprehensive performance characterization.

Overview
--------

Microbenchmarks complement the black-box evaluation by providing more granular insights into system performance. While black-box evaluation measures end-to-end performance through API calls, microbenchmarks can provide direct measurements of the underlying inference engine performance.

Key Features
------------

- **Fine-grained latency measurements**: Precise timing of prefill and decode phases
- **Batch size analysis**: Performance characteristics across different batch configurations
- **Prompt length variation**: Understanding how input length affects latency
- **System comparison**: Detailed metrics for optimization and capacity planning

Use Cases
---------

Microbenchmarks are particularly useful for:

- **Performance optimization**: Identifying bottlenecks in inference pipelines
- **Hardware evaluation**: Comparing different GPU configurations
- **Model comparison**: Analyzing performance differences between model architectures
- **Capacity planning**: Understanding resource utilization patterns

Getting Started
---------------

To run microbenchmarks with ``veeksha``, you'll need access to the underlying inference system for direct performance measurements. Check the main documentation for setup instructions and configuration options.

.. note::
   Microbenchmarking requires direct access to the inference engine and may not be available for all deployment scenarios. For API-only access, use the black-box evaluation approach instead.