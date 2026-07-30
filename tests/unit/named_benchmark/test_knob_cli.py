"""Free-variable CLI peeling and named-run parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from veeksha.named_benchmark.knobs import (
    KnobDeclarationError,
    find_cli_option,
    parse_knob_specs,
    peel_knob_cli_args,
)
from veeksha.cli.free_variables import parse_benchmark_run_configs


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
def test_find_cli_option_forms() -> None:
    assert find_cli_option(["--benchmark", "x", "--y", "1"], "benchmark") == "x"
    assert find_cli_option(["--benchmark=y"], "benchmark") == "y"
    assert find_cli_option(["--other", "1"], "benchmark") is None


@pytest.mark.unit
def test_peel_knob_cli_args() -> None:
    remaining, values = peel_knob_cli_args(
        [
            "--benchmark",
            "local",
            "--concurrency",
            "64",
            "--endpoint.api_base",
            "http://x",
        ],
        _specs(),
    )
    assert values == {"concurrency": 64}
    assert remaining == [
        "--benchmark",
        "local",
        "--endpoint.api_base",
        "http://x",
    ]


@pytest.mark.unit
def test_peel_knob_equals_form() -> None:
    remaining, values = peel_knob_cli_args(["--concurrency=8", "--seed", "1"], _specs())
    assert values == {"concurrency": 8}
    assert remaining == ["--seed", "1"]


@pytest.mark.unit
def test_peel_rejects_bad_choice() -> None:
    with pytest.raises(KnobDeclarationError, match="not one of"):
        peel_knob_cli_args(["--concurrency", "99"], _specs())


@pytest.mark.unit
def test_parse_benchmark_run_with_knobs(tmp_path: Path) -> None:
    def_dir = tmp_path / "bench"
    def_dir.mkdir()
    (def_dir / "benchmark.yml").write_text(
        yaml.safe_dump(
            {
                "name": "unit-bench",
                "knobs": {
                    "concurrency": {
                        "target": "traffic_scheduler.target_concurrent_sessions",
                        "type": "int",
                        "default": 1,
                        "choices": [1, 8, 64],
                    }
                },
                "config": {
                    "seed": 1,
                    "session_generator": {"type": "synthetic"},
                    "traffic_scheduler": {
                        "type": "concurrent",
                        "target_concurrent_sessions": 1,
                    },
                    "client": {"type": "tts"},
                    "evaluators": [{"type": "performance"}],
                },
            }
        ),
        encoding="utf-8",
    )

    configs = parse_benchmark_run_configs(
        [
            "--benchmark",
            str(def_dir),
            "--concurrency",
            "64",
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
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.benchmark == str(def_dir)
    assert getattr(cfg, "_knob_overrides") == {"concurrency": 64}
    provided = getattr(cfg, "_cli_provided_keys")
    assert "concurrency" in provided
    assert "benchmark" in provided
