Capacity Search
===============

Capacity Search is a tool to help find maximal QPS given different SLOs. There are two types of SLOs:

1. **TBT based:** does QPS search based on tbt and ttft slo with their percentiles.
2. **TPOT based:** does QPS search based on ttft and tpot slo with their percentiles.

Below figure shows maximum capacity achieved for different SLOs for Llama-3-8B on different traces and open source systems on H100 GPU:

.. image:: ../_static/assets/capacity_bars.png
    :alt: capacity_bars
    :align: center

Following sections explain running capacity search for each of the above SLOs.

TBT Based SLO
~~~~~~~~~~~~~

.. code-block:: shell

    python -m veeksha.capacity_search.main \
    --output-dir "cap_experiments/capacity_search/" \
    --slo-type tbt_ttft \
    --tbt-slo 0.03 \
    --tbt-percentile 0.9 \
    --ttft-slo 0.3 \
    --ttft-percentile 0.9 \
    --max-iterations 10 \
    --config-path ./veeksha/capacity_search/config/llama_8b.yml

TPOT Based SLO
~~~~~~~~~~~~~~

.. code-block:: shell

    python -m veeksha.capacity_search.main \
    --output-dir "cap_experiments/capacity_search/" \
    --slo-type ttft_tpot \
    --ttft-slo 0.3 \
    --ttft-percentile 0.9 \
    --tpot-slo 0.03 \
    --tpot-percentile 0.9 \
    --max-iterations 10 \
    --config-path ./veeksha/capacity_search/config/llama_8b.yml

Caching
~~~~~~~

The capacity search runs for given model and open source system are cached. This means, when we run capacity search again with different SLO type and values, the benchmark runs with previously explored QPS values will be used directly instead of doing new benchmark runs.

