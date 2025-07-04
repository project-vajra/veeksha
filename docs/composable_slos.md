# Composable SLO Configuration System

## Overview

The composable SLO system provides a flexible and extensible way to define Service Level Objectives (SLOs) for LLM inference systems. This new system addresses the limitations of the previous rigid configuration approach by using a polymorphic design.

## Key Features

1. **Composable SLOs**: Define multiple SLOs that can be evaluated together
2. **Flexible Metrics**: Support for `TTFT`, `TBT`, `TPOT`, and `DEADLINE_MISS_RATE`
3. **Dynamic Thresholds**: SLOs can be constant values, prediction-based multipliers, or offsets
4. **Percentile Control**: Each SLO can specify its own percentile for evaluation
5. **Composition Logic**: Choose whether all SLOs must be met (`require_all: true`) or any single SLO is sufficient (`require_all: false`)

## SLO Definition Structure

Each SLO is defined as a dictionary in a YAML or JSON file. The `type` field determines the kind of SLO.

```yaml
# Basic structure
- type: "constant" # or "prediction_multiplier", "prediction_offset"
  metric: "ttft"
  value: 0.3
  percentile: 0.9
  name: "P90 TTFT"
```

### Common Properties
- `type` (string, required): The type of SLO. Must be one of `constant`, `prediction_multiplier`, `prediction_offset`
- `metric` (string, required): The metric to evaluate. Must be one of `ttft`, `tbt`, `tpot`, `deadline_miss_rate`
- `percentile` (float): The percentile at which to evaluate the metric (e.g., 0.9 for P90). Defaults to `0.99`
- `name` (string): A human-readable name for the SLO

---

## SLO Types

### 1. Constant SLO
A constant SLO has a fixed threshold value

**Type:** `constant`

**Required property:**
- `value` (float): The constant threshold

**Example:**
```yaml
- type: "constant"
  metric: "tbt"
  value: 0.05  # 50ms
  percentile: 0.99
```

### 2. Prediction-Based SLOs
Prediction-based SLOs derive their threshold from a pre-trained performance model (the "prefill predictor"). These SLOs are **only valid for `TTFT`**

#### Prediction Multiplier
The threshold is the predicted TTFT multiplied by a factor

**Type:** `prediction_multiplier`

**Required property:**
- `value` (float): The multiplier

**Optional properties:**
- `predictor_field` (string): The request field used for prediction lookup. Defaults to `"num_total_tokens"`
- `min_value` (float): A minimum clamp for the calculated threshold
- `max_value` (float): A maximum clamp for the calculated threshold

**Example:**
```yaml
- type: "prediction_multiplier"
  metric: "ttft"
  value: 1.5  # Threshold is 1.5x the prediction
  percentile: 0.95
```

#### Prediction Offset
The threshold is the predicted TTFT plus a fixed offset (slack)

**Type:** `prediction_offset`

**Required property:**
- `value` (float): The offset value to add to the prediction

**Optional properties:** Same as Prediction Multiplier

**Example:**
```yaml
- type: "prediction_offset"
  metric: "ttft"
  value: 0.3  # Threshold is prediction + 300ms
  percentile: 0.9
```

---

## Usage with Capacity Search

To use the composable SLO system, create a YAML file (e.g., `slo_config.yaml`) and pass it to the capacity search script

**Example `slo_config.yaml`:**
```yaml
# All of these SLOs must be met
require_all: true
slos:
  - type: "constant"
    metric: "tbt"
    value: 0.05
    percentile: 0.99
    name: "P99 TBT"
  - type: "prediction_offset"
    metric: "ttft"
    value: 0.3
    percentile: 0.9
    name: "P90 Dynamic TTFT"
```

**Run capacity search:**
```bash
python -m veeksha.capacity_search.main \
    --output-dir "experiments/" \
    --composable-slo-config-file "slo_config.yaml" \
    --config-path ./veeksha/capacity_search/config/llama_8b.yml
```

### Backward Compatibility
The system is backward compatible with the legacy SLO flags (`--slo-type`, `--tbt-slo`, etc.). If `composable_slo_config_file` is **not** provided, the script will automatically construct an SLO configuration from the legacy flags

---

## Extending the System

The polymorphic design makes the system easy to extend

### Adding a New Metric
1. Add the new metric to the `SLOMetric` enum in `veeksha/config/slo.py`
2. Update the `SLOEvaluator._extract_metric_values()` method in `veeksha/metrics/slo_evaluator.py` to correctly extract the new metric from the results file

### Adding a New SLO Type
1. Create a new dataclass that inherits from `BaseSLODefinition` in `veeksha/config/slo.py`
2. Implement the `@classmethod def get_type(cls) -> str:` method
3. Implement the `def get_threshold(self, **kwargs) -> float:` method with the new logic
4. If the new type is prediction-based, inherit from `PredictionSLODefinition` to get validation for the `TTFT` metric

## Benefits Over Previous System

1. **Flexibility**: No more hardcoded fields for each metric
2. **Extensibility**: Easy to add new metrics and SLO types
3. **Composability**: Combine multiple SLOs with different evaluation logic
4. **Fine-grained Control**: Each SLO can have its own percentile
5. **Dynamic Behavior**: Support for prediction-based and adaptive SLOs
6. **Backward Compatibility**: Existing configurations continue to work

## Migration Guide

To migrate from the old system to the new composable system:

1. Identify your current SLO configuration
2. Create equivalent `SLODefinition` objects
3. Combine them in a `ComposableSLOConfig`
4. Use the new configuration in your capacity search

Example migration:
```python
# Old way
capacity_config = CapacitySearchConfig(
    slo_type="tbt_ttft",
    tbt_slo=0.03,
    tbt_percentile=0.9,
    ttft_slo=0.3,
    ttft_percentile=0.9
)

# New way
slo_config = ComposableSLOConfig(
    slos=[
        SLODefinition(metric=SLOMetric.TBT, value=0.03, percentile=0.9),
        SLODefinition(metric=SLOMetric.TTFT, value=0.3, percentile=0.9),
    ]
) 