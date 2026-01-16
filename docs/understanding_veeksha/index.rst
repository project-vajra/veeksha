Understanding Veeksha
=====================

This section provides a deep dive into how Veeksha works internally. Understanding
these concepts will help you make informed decisions when configuring benchmarks
and interpreting results.

Veeksha is designed around a modular architecture where components can be combined
flexibly to model different workload patterns. The key insight is that LLM workloads
are fundamentally **session-based**: a user conversation consists of multiple
request-response turns, and benchmarking tools should capture this reality.


Overview
--------

At the highest level, Veeksha:

1. **Generates sessions** representing user conversations or agentic flows as directed acyclic graphs (DAGs)
2. **Schedules sessions** according to configurable traffic patterns
3. **Sends requests** to LLM inference endpoints via async HTTP clients
4. **Evaluates results** by computing latency/accuracy metrics and checking SLO compliance


Key design principles
---------------------

**Sessions, not just requests**
    Unlike simple load generators that fire independent requests, Veeksha models
    multi-turn conversations where later requests depend on earlier ones.

**Configurable content generation**
    Request content can be synthetic (random tokens with controlled lengths) or
    trace-driven (replay of real conversation data).

**Flexible traffic patterns**
    Support for both rate-based (Poisson, gamma) and concurrency-based scheduling
    allows testing different operational scenarios.

**Comprehensive metrics**
    Fine-grained timing captures every stage of the request lifecycle, enabling
    detailed performance analysis.


In this section
---------------

.. toctree::
   :maxdepth: 2

   architecture
   sessions_and_graphs
   content_generation
   scheduling
   evaluation