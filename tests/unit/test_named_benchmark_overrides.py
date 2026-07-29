"""Named benchmarks reject overrides of frozen definition fields."""

from __future__ import annotations

import pytest

from veeksha.benchmark_knobs import parse_knob_specs
from veeksha.benchmark_resolve import NamedBenchmarkError, reject_frozen_overrides


def _specs():
    return parse_knob_specs(
        {
            "concurrency": {
                "target": "traffic_scheduler.target_concurrent_sessions",
                "type": "int",
                "default": 1,
                "choices": [1, 8, 64],
            }
        }
    )


@pytest.mark.unit
def test_allows_endpoint_and_identity() -> None:
    reject_frozen_overrides(
        frozenset(
            {
                "benchmark",
                "benchmark_revision",
                "endpoint.api_base",
                "endpoint.model",
                "output_dir",
            }
        ),
        _specs(),
        allow_config_override=False,
    )


@pytest.mark.unit
def test_allows_declared_free_variable_target() -> None:
    reject_frozen_overrides(
        frozenset(
            {
                "benchmark",
                "traffic_scheduler.target_concurrent_sessions",
                "concurrency",
            }
        ),
        _specs(),
        allow_config_override=False,
    )


@pytest.mark.unit
def test_rejects_frozen_seed() -> None:
    with pytest.raises(NamedBenchmarkError, match="seed"):
        reject_frozen_overrides(
            frozenset({"benchmark", "seed"}),
            _specs(),
            allow_config_override=False,
        )


@pytest.mark.unit
def test_rejects_frozen_session_generator() -> None:
    with pytest.raises(NamedBenchmarkError, match="session_generator"):
        reject_frozen_overrides(
            frozenset({"benchmark", "session_generator.type"}),
            _specs(),
            allow_config_override=False,
        )


@pytest.mark.unit
def test_allow_config_override_skips_check() -> None:
    reject_frozen_overrides(
        frozenset({"benchmark", "seed", "session_generator.type"}),
        _specs(),
        allow_config_override=True,
    )
