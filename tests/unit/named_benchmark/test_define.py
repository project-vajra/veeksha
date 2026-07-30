"""Unit tests for named-benchmark definition authoring helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from veeksha.named_benchmark.define import BenchmarkDefineError, define_benchmark
from veeksha.named_benchmark.resolve import expected_fingerprint
from veeksha.config.benchmark_define import BenchmarkDefineConfig
from veeksha.core.workload_fingerprint import WorkloadFingerprint


def _write_minimal_definition(root: Path, *, max_sessions: int = 4) -> Path:
    """Write a definition whose config is only used for pin bookkeeping."""
    config = {
        "seed": 7,
        "runtime": {"max_sessions": max_sessions},
        "session_generator": {"type": "synthetic"},
        "traffic_scheduler": {
            "type": "concurrent",
            "target_concurrent_sessions": 1,
        },
        "client": {"type": "tts"},
        "evaluators": [{"type": "performance"}],
    }
    definition = {
        "name": "unit-synthetic",
        "version": 1,
        "purpose": "Unit-test synthetic workload.",
        "knobs": {
            "concurrency": {
                "target": "traffic_scheduler.target_concurrent_sessions",
                "type": "int",
                "default": 1,
                "choices": [1, 2, 4],
                "help": "Concurrency (must leave the workload unchanged).",
            }
        },
        "config": config,
    }
    path = root / "benchmark.yml"
    path.write_text(yaml.safe_dump(definition), encoding="utf-8")
    return path


class _FakeFingerprint(WorkloadFingerprint):
    def __init__(self, digest: str, sessions: int = 3) -> None:
        super().__init__()
        self._forced = digest
        self._session_count = sessions
        self._request_count = sessions

    def digest(self) -> str:
        return self._forced


def _install_fingerprint_stub(
    monkeypatch: pytest.MonkeyPatch, digest_for_target: dict[str, str]
) -> None:
    """Stub generation so fingerprint can depend on concurrency for tests."""

    def fake_generate(
        benchmark_config: Any, *, max_sessions: int
    ) -> WorkloadFingerprint:
        target = getattr(
            benchmark_config.traffic_scheduler, "target_concurrent_sessions", None
        )
        digest = digest_for_target.get(
            str(target), digest_for_target.get("*", "blake2b:default")
        )
        return _FakeFingerprint(digest, sessions=max_sessions)

    monkeypatch.setattr("veeksha.named_benchmark.define._generate_fingerprint", fake_generate)

    class _Tok:
        model_name = "stub-model"

    monkeypatch.setattr(
        "veeksha.config.client.TTSClientConfig.build_tokenizer_provider",
        lambda self: _Tok(),
    )


@pytest.mark.unit
def test_define_benchmark_writes_single_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yml = _write_minimal_definition(tmp_path)
    # Free variable is pin-stable: all concurrency values share one digest.
    _install_fingerprint_stub(
        monkeypatch,
        {
            "*": "blake2b:shared",
            "1": "blake2b:shared",
            "2": "blake2b:shared",
            "4": "blake2b:shared",
        },
    )

    result = define_benchmark(BenchmarkDefineConfig(definition=str(yml)))
    assert result["name"] == "unit-synthetic"
    pins = result["pins"]
    assert pins["workload_fingerprint"] == "blake2b:shared"
    assert pins["sessions_sampled"] == 4  # from config.runtime.max_sessions
    assert pins["knob_defaults"] == {"concurrency": 1}
    assert (tmp_path / "pins.json").is_file()
    reloaded = yaml.safe_load((tmp_path / "benchmark.yml").read_text(encoding="utf-8"))
    assert reloaded["pins"]["workload_fingerprint"] == "blake2b:shared"
    assert expected_fingerprint({"pins": pins}) == "blake2b:shared"


@pytest.mark.unit
def test_define_rejects_free_variable_that_moves_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yml = _write_minimal_definition(tmp_path)
    _install_fingerprint_stub(
        monkeypatch,
        {"1": "blake2b:a", "2": "blake2b:b", "4": "blake2b:c"},
    )
    with pytest.raises(BenchmarkDefineError, match="changes the workload fingerprint"):
        define_benchmark(BenchmarkDefineConfig(definition=str(yml)))


@pytest.mark.unit
def test_expected_fingerprint_reads_string_pin() -> None:
    assert (
        expected_fingerprint({"pins": {"workload_fingerprint": "blake2b:pin"}})
        == "blake2b:pin"
    )
    assert expected_fingerprint({"pins": {}}) is None


@pytest.mark.unit
def test_define_requires_positive_max_sessions_in_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yml = _write_minimal_definition(tmp_path, max_sessions=-1)
    _install_fingerprint_stub(monkeypatch, {"*": "blake2b:x", "1": "blake2b:x", "2": "blake2b:x", "4": "blake2b:x"})
    with pytest.raises(BenchmarkDefineError, match="runtime.max_sessions"):
        define_benchmark(BenchmarkDefineConfig(definition=str(yml)))
