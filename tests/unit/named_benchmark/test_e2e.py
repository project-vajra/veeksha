"""End-to-end named-benchmark path without requiring transformers.

Uses a deterministic session generator so define → pin → resolve → preflight
runs fully offline. Covers the real fingerprint, define, and pin-check code.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from veeksha.named_benchmark.define import BenchmarkDefineError, define_benchmark
from veeksha.named_benchmark.resolve import (
    NamedBenchmarkError,
    check_workload_pin,
    resolve_named_benchmark,
)
from veeksha.config.benchmark_define import BenchmarkDefineConfig
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.session import Session
from veeksha.core.session_graph import SessionGraph, SessionNode, add_node
from veeksha.core.workload_fingerprint import WorkloadFingerprint
from veeksha.types import ChannelModality


class _FixedSessionGenerator:
    """Yields a fixed sequence of sessions; independent of tokenizer/HF."""

    def __init__(self, n: int = 4, text: str = "hello world") -> None:
        self._n = n
        self._text = text
        self._i = 0

    def generate_session(self) -> Session:
        if self._i >= self._n:
            raise StopIteration
        sid = self._i
        self._i += 1
        graph = SessionGraph()
        add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
        req = Request(
            id=sid * 10,
            channels={
                ChannelModality.TEXT: TextChannelRequestContent(
                    input_text=f"{self._text} {sid}"
                )
            },
            metadata={"idx": sid},
        )
        return Session(id=sid, session_graph=graph, requests={0: req})


def _write_definition(root: Path) -> Path:
    config = {
        "seed": 7,
        "runtime": {"max_sessions": 4, "pregenerate_sessions": True},
        "session_generator": {"type": "synthetic"},
        "traffic_scheduler": {
            "type": "concurrent",
            "target_concurrent_sessions": 1,
        },
        "client": {"type": "tts"},
        "evaluators": [{"type": "performance"}],
    }
    definition = {
        "name": "e2e-fixed",
        "version": 1,
        "purpose": "Offline e2e named benchmark.",
        "knobs": {
            "concurrency": {
                "target": "traffic_scheduler.target_concurrent_sessions",
                "type": "int",
                "default": 1,
                "choices": [1, 2, 4],
                "help": "Load only.",
            }
        },
        "config": config,
    }
    path = root / "benchmark.yml"
    path.write_text(yaml.safe_dump(definition), encoding="utf-8")
    return path


def _install_fixed_generator(
    monkeypatch: pytest.MonkeyPatch, text: str = "hello"
) -> None:
    def fake_get(key, **kwargs):
        return _FixedSessionGenerator(n=4, text=text)

    monkeypatch.setattr(
        "veeksha.named_benchmark.define.SessionGeneratorRegistry.get",
        fake_get,
    )

    class _Tok:
        model_name = "fixed-stub"

    monkeypatch.setattr(
        "veeksha.config.client.TTSClientConfig.build_tokenizer_provider",
        lambda self: _Tok(),
    )


@pytest.mark.unit
def test_define_resolve_preflight_pin_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yml = _write_definition(tmp_path)
    _install_fixed_generator(monkeypatch, text="hello")

    result = define_benchmark(BenchmarkDefineConfig(definition=str(yml)))
    pin = result["pins"]["workload_fingerprint"]
    assert pin.startswith("blake2b:")
    assert result["pins"]["sessions_sampled"] == 4

    # Re-resolve like a run would, with a free-variable override.
    from veeksha.cli.free_variables import parse_benchmark_run_configs

    configs = parse_benchmark_run_configs(
        [
            "--benchmark",
            str(tmp_path),
            "--concurrency",
            "4",
            "--endpoint.engine_type",
            "vllm",
            "--endpoint.api_base",
            "http://127.0.0.1:9/v1",
            "--endpoint.model",
            "m",
            "--output_dir",
            str(tmp_path / "out"),
        ]
    )
    resolved, meta = resolve_named_benchmark(
        configs[0], knob_overrides=getattr(configs[0], "_knob_overrides", None)
    )
    assert meta["knobs"]["concurrency"] == 4
    assert resolved.traffic_scheduler.target_concurrent_sessions == 4
    expected_pin = meta["pins"]["workload_fingerprint"]
    assert expected_pin == pin

    # Preflight: same generator sequence → same fingerprint → OK
    fp = WorkloadFingerprint()
    gen = _FixedSessionGenerator(n=4, text="hello")
    for _ in range(4):
        fp.add_session(gen.generate_session())
    check_workload_pin(
        actual_digest=fp.digest(),
        named_meta=meta,
        allow_workload_drift=False,
        stage="preflight",
    )
    assert fp.digest() == pin


@pytest.mark.unit
def test_preflight_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yml = _write_definition(tmp_path)
    _install_fixed_generator(monkeypatch, text="hello")
    result = define_benchmark(BenchmarkDefineConfig(definition=str(yml)))
    pin = result["pins"]["workload_fingerprint"]
    meta = {
        "name": "e2e-fixed",
        "pins": {"workload_fingerprint": pin},
    }

    # Different stream text → different fingerprint
    fp = WorkloadFingerprint()
    gen = _FixedSessionGenerator(n=4, text="DIFFERENT")
    for _ in range(4):
        fp.add_session(gen.generate_session())
    assert fp.digest() != pin
    with pytest.raises(NamedBenchmarkError, match="preflight"):
        check_workload_pin(
            actual_digest=fp.digest(),
            named_meta=meta,
            allow_workload_drift=False,
            stage="preflight",
        )


@pytest.mark.unit
def test_define_rejects_free_var_that_moves_real_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a free var actually changes generation, define must fail."""
    yml = _write_definition(tmp_path)
    n_calls = {"i": 0}

    def fake_get(key, **kwargs):
        n_calls["i"] += 1
        # call 1 = base pin; call 2+ = free-var alternate verification
        text = "base" if n_calls["i"] == 1 else f"alt-{n_calls['i']}"
        return _FixedSessionGenerator(n=2, text=text)

    monkeypatch.setattr(
        "veeksha.named_benchmark.define.SessionGeneratorRegistry.get", fake_get
    )

    class _Tok:
        model_name = "fixed-stub"

    monkeypatch.setattr(
        "veeksha.config.client.TTSClientConfig.build_tokenizer_provider",
        lambda self: _Tok(),
    )

    with pytest.raises(BenchmarkDefineError, match="changes the workload fingerprint"):
        define_benchmark(BenchmarkDefineConfig(definition=str(yml)))
