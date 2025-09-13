# Capacity Search

Tool to help find maximal QPS given different SLOs. There are two types of SLOs:
1. TBT-TTFT based: does QPS search based on tbt and ttft slo with their percentiles.
2. TTFT-TPOT based: does QPS search based on ttft and tpot slo with their percentiles.

## TBT-TTFT based Capacity Search
```bash
python -m veeksha.capacity_search \
--output-dir "cap_experiments/capacity_search/" \
--slo-type tbt_ttft \
--tbt-slo 0.03 \
--tbt-percentile 0.9 \
--ttft-slo 0.3 \
--ttft-percentile 0.9 \
--max-iterations 10 \
--config-path ./veeksha/capacity_search/config/llama_8b.yml
```

## TTFT-TPOT based Capacity Search
```bash
python -m veeksha.capacity_search \
--output-dir "cap_experiments/capacity_search/" \
--slo-type ttft_tpot \
--ttft-slo 0.3 \
--ttft-percentile 0.9 \
--tpot-slo 0.03 \
--tpot-percentile 0.9 \
--max-iterations 10 \
--config-path ./veeksha/capacity_search/config/llama_8b.yml
```

## Caching
The capacity search runs for given model and open source system are cached. This means, when we run capacity search again with different SLO type and values, the benchmark runs with previously explored QPS values will be used directly instead of doing new benchmark runs.
