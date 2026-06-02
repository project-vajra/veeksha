"""YAML and template helpers for sweep planning."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import yaml

from veeksha.sweeps.config import TRACE_SEED_TTS, TRACE_SHAREGPT, SweepConfig
from veeksha.sweeps.specs import INPUT_SWEEP, SweepSpec


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Base config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Base config must be a YAML mapping: {path}")
    return data


def write_config(path: Path, config: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, width=1_000_000)


def mapping_at(config: Dict[str, Any], path: Sequence[str]) -> Dict[str, Any]:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            dotted = ".".join(path)
            raise KeyError(f"Missing YAML mapping: {dotted}")
        node = node[key]
    if not isinstance(node, dict):
        dotted = ".".join(path)
        raise TypeError(f"Expected YAML mapping at {dotted}")
    return node


def set_required(config: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    parent = mapping_at(config, path[:-1])
    key = path[-1]
    if key not in parent:
        raise KeyError(f"Missing YAML key: {'.'.join(path)}")
    parent[key] = value


def apply_trace_source(config: Dict[str, Any], sweep_config: SweepConfig) -> None:
    session_generator = mapping_at(config, ("session_generator",))
    if session_generator.get("type") != "trace":
        raise ValueError(
            "--trace can only be used with trace session generator configs"
        )

    flavor = session_generator.get("flavor")
    if not isinstance(flavor, dict):
        raise TypeError("Expected YAML mapping at session_generator.flavor")

    if sweep_config.trace == TRACE_SHAREGPT:
        flavor["type"] = "sharegpt"
        if "assistant_role" not in flavor:
            flavor["assistant_role"] = "gpt"
        return

    if sweep_config.trace != TRACE_SEED_TTS:
        raise ValueError(f"Unsupported trace source: {sweep_config.trace}")

    seed_flavor = {"type": "seed_tts_text"}
    for key in ("min_tokens", "max_tokens", "min_chars", "max_chars"):
        if key in flavor:
            seed_flavor[key] = flavor[key]

    session_generator["trace_file"] = ""
    session_generator["flavor"] = seed_flavor


def format_template(
    template: str, *, concurrency: int, input_size: Optional[int] = None
) -> str:
    return template.format(
        concurrency=concurrency,
        input_size=input_size,
        date_tag=_date_tag(),
    )


def input_sizes(sweep_config: SweepConfig, spec: SweepSpec) -> Tuple[int, ...]:
    if sweep_config.sizes:
        return sweep_config.sizes

    start = (
        sweep_config.range_start
        if sweep_config.range_start is not None
        else spec.default_range_start
    )
    end = (
        sweep_config.range_end
        if sweep_config.range_end is not None
        else spec.default_range_end
    )
    step = sweep_config.step if sweep_config.step is not None else spec.default_step
    if start is None or end is None or step is None:
        raise ValueError("Input sweep requires --sizes or range defaults")
    if start <= 0 or end <= 0 or step <= 0:
        raise ValueError("--range-start, --range-end, and --step must be positive")
    if start > end:
        raise ValueError("--range-start must be <= --range-end")
    return tuple(range(start, end + 1, step))


def build_run_config(
    base_config: Dict[str, Any],
    spec: SweepSpec,
    *,
    concurrency: int,
    input_size: Optional[int],
    timeout_seconds: int,
    max_sessions: int,
    output_dir_template: Optional[str],
    min_tokens: Optional[int],
    max_tokens: Optional[int],
    min_chars: Optional[int],
    max_chars: Optional[int],
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)

    set_required(
        config,
        ("traffic_scheduler", "target_concurrent_sessions"),
        concurrency,
    )
    set_required(config, ("runtime", "num_client_threads"), concurrency)
    if spec.write_runtime_limits:
        set_required(config, ("runtime", "benchmark_timeout"), timeout_seconds)
        set_required(config, ("runtime", "max_sessions"), max_sessions)

    run_name = format_template(
        spec.run_name_template,
        concurrency=concurrency,
        input_size=input_size,
    )
    set_required(config, ("wandb", "run_name"), run_name)

    if output_dir_template:
        config["output_dir"] = _format_output_dir(
            output_dir_template,
            spec,
            concurrency=concurrency,
            input_size=input_size,
            run_name=run_name,
        )

    if spec.sweep_type == INPUT_SWEEP:
        if input_size is None:
            raise ValueError("input_size is required for input sweeps")
        flavor = mapping_at(config, ("session_generator", "flavor"))
        _clear_length_bounds(flavor)
        flavor["min_chars"] = input_size
        flavor["max_chars"] = input_size
        if spec.disable_audio_for_input:
            _disable_audio_saving(config)
    else:
        _apply_length_bounds(
            config,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            min_chars=min_chars,
            max_chars=max_chars,
        )

    return config


def benchmark_command(config_path: Path) -> list[str]:
    return [
        "python",
        "-Xgil=0",
        "-m",
        "veeksha.benchmark",
        "--benchmark-config-from-file",
        str(config_path),
    ]


def override_client_api_base(config: Dict[str, Any], api_base: Optional[str]) -> None:
    if api_base is None:
        return
    client = mapping_at(config, ("client",))
    client["api_base"] = api_base


def _date_tag() -> str:
    return datetime.now().strftime("%m_%d")


def _format_output_dir(
    template: str,
    spec: SweepSpec,
    *,
    concurrency: int,
    input_size: Optional[int],
    run_name: str,
) -> str:
    try:
        return template.format(
            concurrency=concurrency,
            input_size=input_size,
            run_name=run_name,
            date_tag=_date_tag(),
            engine=spec.engine,
            model=spec.model,
            sweep_type=spec.sweep_type,
        )
    except KeyError as exc:
        raise ValueError(f"Unknown --output-dir template field: {exc.args[0]}") from exc


def _disable_audio_saving(config: Dict[str, Any]) -> None:
    evaluators = config.get("evaluators")
    if not isinstance(evaluators, list):
        return
    for evaluator in evaluators:
        if not isinstance(evaluator, dict):
            continue
        if evaluator.get("type") == "audio_quality" and "save_audio_files" in evaluator:
            evaluator["save_audio_files"] = False


def _clear_length_bounds(flavor: Dict[str, Any]) -> None:
    for key in ("min_tokens", "max_tokens", "min_chars", "max_chars"):
        flavor.pop(key, None)


def _apply_length_bounds(
    config: Dict[str, Any],
    *,
    min_tokens: Optional[int],
    max_tokens: Optional[int],
    min_chars: Optional[int],
    max_chars: Optional[int],
) -> None:
    flavor = mapping_at(config, ("session_generator", "flavor"))
    _clear_length_bounds(flavor)

    if min_chars is not None and max_chars is not None:
        flavor["min_chars"] = min_chars
        flavor["max_chars"] = max_chars
        return

    if min_tokens is None or max_tokens is None:
        raise ValueError("min_tokens and max_tokens must be set together")
    flavor["min_tokens"] = min_tokens
    flavor["max_tokens"] = max_tokens
