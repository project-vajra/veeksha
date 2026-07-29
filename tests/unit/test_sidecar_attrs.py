"""Named-benchmark runtime state must survive every config rebuild.

``_named_benchmark_meta`` is attached to a frozen ``BenchmarkConfig`` with
``object.__setattr__``, so it is not a dataclass field and ``replace()`` drops
it. When that happened in ``_with_endpoint`` the pre-flight workload pin check
read an empty meta and silently skipped -- a named benchmark ran against an
unverified workload and still exited 0. These pin the rebuild paths.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from veeksha.benchmark import _with_endpoint
from veeksha.config.benchmark import BenchmarkConfig, carry_sidecar_attrs
from veeksha.config.endpoint import EndpointConfig

META = {"name": "demo", "pins": {"workload_fingerprint": "blake2b:abc"}}


def _named_config(**overrides) -> BenchmarkConfig:
    cfg = BenchmarkConfig(benchmark="demo", **overrides)
    object.__setattr__(cfg, "_named_benchmark_meta", META)
    object.__setattr__(cfg, "_knob_overrides", {"concurrency": 4})
    object.__setattr__(cfg, "_cli_provided_keys", frozenset({"concurrency"}))
    return cfg


@pytest.mark.unit
def test_plain_replace_drops_sidecar_state() -> None:
    """Documents the failure mode the helper exists to prevent."""
    cfg = _named_config()
    assert getattr(
        replace(cfg, output_dir="/tmp/x"), "_named_benchmark_meta", None
    ) is (None)


@pytest.mark.unit
def test_carry_sidecar_attrs_restores_all_state() -> None:
    cfg = _named_config()
    rebuilt = carry_sidecar_attrs(cfg, replace(cfg, output_dir="/tmp/x"))

    assert rebuilt._named_benchmark_meta == META
    assert rebuilt._knob_overrides == {"concurrency": 4}
    assert rebuilt._cli_provided_keys == frozenset({"concurrency"})


@pytest.mark.unit
def test_with_endpoint_preserves_named_benchmark_meta() -> None:
    """The endpoint rebuild is the documented way to run a named benchmark."""
    endpoint = EndpointConfig(
        engine_type="vllm", model="gpt2", api_base="http://127.0.0.1:8099/v1"
    )
    cfg = _named_config(endpoint=endpoint)

    rebuilt = _with_endpoint(cfg, endpoint)

    # Empty meta is what disables the pin check, so assert the value itself.
    assert getattr(rebuilt, "_named_benchmark_meta", None) == META
    assert rebuilt.client.api_base == "http://127.0.0.1:8099/v1"


@pytest.mark.unit
def test_carry_sidecar_attrs_is_a_noop_for_plain_configs() -> None:
    cfg = BenchmarkConfig()
    rebuilt = carry_sidecar_attrs(cfg, replace(cfg, output_dir="/tmp/x"))

    assert getattr(rebuilt, "_named_benchmark_meta", None) is None
    assert rebuilt.output_dir == "/tmp/x"
