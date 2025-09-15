How to use veeksha
=================

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/blackbox_evaluation
   tutorials/measuring_qps
   tutorials/microbenchmarks

``veeksha`` can evaluate LLM inference systems as a black-box and also determine their serving capacity.

``veeksha`` provides three evaluation recipes as described below:

- **Black-box Evaluation**: ``veeksha`` hits LLM inference server exposed through API endpoint with a set of requests with different prompt lengths and tracks when each output token is generated. This allows ``veeksha`` to calculate several metrics like TTFT, TBT, TPOT, and normalized latency to provide comprehensive performance analysis.

- **Capacity Evaluation**: When deploying an LLM inference system, the operator needs to know how many requests can be served by the system. This will help operator in determining the configuration of the system, for example, the number of GPUs needed, to meet certain service quality requirements. To help with this process, ``veeksha`` provides a capacity evaluation module which determines maximum capacity each replica can provide under different request loads while meeting target SLO requirements.

- **Microbenchmarks**: ``veeksha`` provides specialized performance profiling tools to measure the latency characteristics of LLM inference systems. The microbenchmarking system focuses on two critical phases: prefill latency (time-to-first-token) and decode throughput (time-between-tokens) across different prompt lengths and batch sizes. This enables detailed performance characterization and system comparison for optimization and capacity planning.


The description of each metric used in black-box evaluation is provided in :doc:`tutorials/metrics_used`.

Check out the following resources to learn more:

* :doc:`tutorials/blackbox_evaluation`
* :doc:`tutorials/measuring_qps`
* :doc:`tutorials/microbenchmarks`
* :doc:`tutorials/metrics_used`
