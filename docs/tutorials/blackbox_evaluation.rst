Black-box Evaluation
====================

``veeksha`` performs black-box evaluation of both proprietary and open-source systems.

Check out the following resources to learn how to run ``veeksha`` with both proprietary and open-source systems:

.. toctree::
    :maxdepth: 2

    public_apis
    open_source_systems

Running Benchmarks Against Multiple Endpoints in Parallel
----------------------------------------------------------

``veeksha`` supports running benchmarks against multiple API endpoints simultaneously. This is useful for comparing different inference systems, testing load balancing setups, or benchmarking multiple model deployments concurrently.

Using Configuration Files
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can specify multiple endpoints using YAML configuration files with a list of ``endpoint`` configurations:

.. code-block:: yaml

    # example_parallel.yml
    endpoint:
      - name: local-A
        api_url: http://localhost:30000/v1
        api_key: token-abc123
      - name: local-B
        api_url: http://localhost:30002/v1
        api_key: token-def456
      - name: local-C
        api_url: http://localhost:30004/v1
        api_key: token-ghi789

    client_config:
      model: meta-llama/Meta-Llama-3-8B-Instruct
      num_clients: 15
      ...

Run the benchmark with:

.. code-block:: bash

    python -m veeksha.benchmark --config veeksha/benchmark_config_files/example_parallel.yml

How It Works
^^^^^^^^^^^^

- **Endpoint Configuration**: Each endpoint is defined with its ``api_url``, ``api_key``, and optional ``name`` for identification in logs and results.
- **Parallel Execution**: Benchmarks are grouped by endpoint and executed in parallel using separate processes. Each endpoint runs its benchmark configurations sequentially, but different endpoints run simultaneously.
- **List Expansion**: The list of endpoints naturally expands via the config system's list explosion mechanism, creating separate benchmark runs for each endpoint.
- **Other List Fields**: Any other configuration fields specified as lists will expand via cartesian product as usual, creating multiple benchmark runs per endpoint.

Following figures show evaluations by ``veeksha``:

.. _token_rate_comparison_api:

.. figure:: ../_static/assets/token_rate_comparison_api.png
    :alt: toke_rate_comparison_api
    :align: center
    
    **Token Rate Comparison**

Above figure depicts throughput measured by ``veeksha`` for different systems based on three different metrics:

* TPOT
* TBT
* *fluid-token-generation-rate*: Here we find minimum TBT latency such that 99% of requests have *fluidity-index* at least 0.9. Inverse of TBT latency is *fluid-token-generation-rate*.

.. _tbt_cdf_api:

.. figure:: ../_static/assets/tbt_cdf_api_1.png
    :alt: tbt_cdf_api
    :align: center
    
    **TBT CDF**

Above figure depicts TBT CDF for different systems. It is difficult to interpret the difference in TBT across different systems.

.. _tbt_acceptance_rate_curve:

.. figure:: ../_static/assets/tbt_acceptance_rate_curve.png
    :alt: tbt_acceptance_rate_curve
    :align: center
    
    **TBT Acceptance Rate Curve**

Above figure clearly highlights the difference in TBT across different systems which was difficult to interpret in previous figure, :ref:`tbt_cdf_api`.
