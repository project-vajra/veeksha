"""Example configurations for the composable SLO system."""

from veeksha.config.slo import (
    ComposableSLOConfig,
    ConstantSLODefinition,
    PredictionMultiplierSLODefinition,
    PredictionOffsetSLODefinition,
    SLOMetric,
)


def create_basic_latency_slos():
    """Create basic latency SLOs with constant thresholds."""
    return ComposableSLOConfig(
        slos=[
            ConstantSLODefinition(
                metric=SLOMetric.TTFT,
                value=0.3,  # 300ms
                percentile=0.9,
                name="P90 TTFT",
            ),
            ConstantSLODefinition(
                metric=SLOMetric.TBT,
                value=0.05,  # 50ms
                percentile=0.99,
                name="P99 TBT",
            ),
        ],
        require_all=True,
    )


def create_dynamic_ttft_slo():
    """Create dynamic TTFT SLO based on predictions."""
    return ComposableSLOConfig(
        slos=[
            PredictionOffsetSLODefinition(
                metric=SLOMetric.TTFT,
                value=0.3,  # 300ms slack added to prediction
                percentile=0.9,
                name="Dynamic TTFT with 300ms slack",
            ),
            ConstantSLODefinition(
                metric=SLOMetric.TBT, value=0.05, percentile=0.99, name="P99 TBT"
            ),
        ],
        require_all=True,
    )


def create_proportional_slos():
    """Create SLOs with prediction multipliers."""
    return ComposableSLOConfig(
        slos=[
            PredictionMultiplierSLODefinition(
                metric=SLOMetric.TTFT,
                value=1.5,  # 1.5x the predicted time
                percentile=0.95,
                min_value=0.1,  # At least 100ms
                max_value=2.0,  # At most 2s
                name="TTFT within 1.5x prediction",
            ),
            ConstantSLODefinition(
                metric=SLOMetric.TPOT, value=0.1, percentile=0.9, name="P90 TPOT"
            ),
        ],
        require_all=True,
    )


def create_mixed_percentile_slos():
    """Create SLOs with different percentiles for different metrics."""
    return ComposableSLOConfig(
        slos=[
            # Tight SLO for median latency
            ConstantSLODefinition(
                metric=SLOMetric.TTFT,
                value=0.2,
                percentile=0.5,  # Median
                name="Median TTFT",
            ),
            # Looser SLO for tail latency
            ConstantSLODefinition(
                metric=SLOMetric.TTFT,
                value=1.0,
                percentile=0.99,  # P99
                name="P99 TTFT",
            ),
        ],
        require_all=True,
    )


def create_any_of_slos():
    """Create SLOs where any one needs to be satisfied."""
    return ComposableSLOConfig(
        slos=[
            # Either fast TTFT
            ConstantSLODefinition(
                metric=SLOMetric.TTFT,
                value=0.1,
                percentile=0.9,
                name="Fast TTFT option",
            ),
            # OR low deadline miss rate
            ConstantSLODefinition(
                metric=SLOMetric.DEADLINE_MISS_RATE,
                value=0.05,  # 5% miss rate
                percentile=0.99,
                name="Low deadline miss rate option",
            ),
        ],
        require_all=False,  # Any SLO can be met
    )


def create_comprehensive_slos():
    """Create a comprehensive set of SLOs for production use."""
    return ComposableSLOConfig(
        slos=[
            PredictionOffsetSLODefinition(
                metric=SLOMetric.TTFT,
                value=0.2,  # 200ms slack
                percentile=0.9,
                name="P90 Dynamic TTFT",
            ),
            ConstantSLODefinition(
                metric=SLOMetric.TBT,
                value=0.03,  # 30ms
                percentile=0.95,
                name="P95 TBT",
            ),
            ConstantSLODefinition(
                metric=SLOMetric.TPOT,
                value=0.05,  # 50ms
                percentile=0.9,
                name="P90 TPOT",
            ),
            ConstantSLODefinition(
                metric=SLOMetric.DEADLINE_MISS_RATE,
                value=0.1,  # 10% miss rate
                percentile=0.99,
                name="P99 Deadline miss rate",
            ),
        ],
        require_all=True,
    )


# Example of how to extend with custom metrics
def create_custom_metric_slo():
    """Example of how to use custom metrics (requires extending SLOMetric enum)."""
    # This would require adding new metrics to SLOMetric enum
    # For example: SLOMetric.GPU_UTILIZATION, SLOMetric.MEMORY_USAGE, etc.
    pass 