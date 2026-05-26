#!/usr/bin/env python3
"""Run Veeksha benchmark sweeps from one launcher.

Examples
--------
    # Concurrency sweep
    python scripts/sweep.py --sweep-type concurrency --engine vajra --model qwen-tts

    # Input-size sweep at fixed concurrency
    python scripts/sweep.py --sweep-type input --engine vllm --model qwen-tts \\
        --concurrency 16 --sizes 20,60,100,200,500

    # Preview generated configs and benchmark commands without running them
    python scripts/sweep.py --sweep-type concurrency --engine vllm \\
        --model qwen3-omni --concurrencies 1,2 --dry-run

    # Use the Seed TTS text dataset instead of the default ShareGPT trace
    python scripts/sweep.py --sweep-type concurrency --engine vajra \\
        --model qwen-tts --trace seed_tts
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

CONCURRENCY_SWEEP = "concurrency"
INPUT_SWEEP = "input"

TRACE_SHAREGPT = "sharegpt"
TRACE_SEED_TTS = "seed_tts"
TRACE_ALIASES = {
    "sharegpt": TRACE_SHAREGPT,
    "share_gpt": TRACE_SHAREGPT,
    "seed_tts": TRACE_SEED_TTS,
    "seedtts": TRACE_SEED_TTS,
    "seed_tts_text": TRACE_SEED_TTS,
}

MODEL_ALIASES = {
    "qwen-tts": "qwen-tts",
    "qwen3-tts": "qwen-tts",
    "tts": "qwen-tts",
    "qwen3-omni": "qwen3-omni",
    "omni": "qwen3-omni",
    "vibe-voice": "vibe-voice",
    "vibe": "vibe-voice",
}


@dataclass(frozen=True)
class SweepSpec:
    sweep_type: str
    engine: str
    model: str
    config_name: str
    temp_prefix: str
    run_config_template: str
    run_name_template: str
    default_concurrencies: Tuple[int, ...] = ()
    default_concurrency: Optional[int] = None
    default_range_start: Optional[int] = None
    default_range_end: Optional[int] = None
    default_step: Optional[int] = None
    write_runtime_limits: bool = True
    disable_audio_for_input: bool = False

    @property
    def config_path(self) -> Path:
        return CONFIG_DIR / self.config_name


SPECS: Dict[Tuple[str, str, str], SweepSpec] = {
    (CONCURRENCY_SWEEP, "vajra", "qwen-tts"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vajra",
        model="qwen-tts",
        config_name="vajra.yaml",
        temp_prefix="vajra_qwen_tts_sweep",
        run_config_template="vajra_qwen_c{concurrency}.yaml",
        run_name_template="vajra_qwen3tts_c_{concurrency}_10_minutes",
        default_concurrencies=(1, 2, 4, 8, 16, 32, 64),
    ),
    (CONCURRENCY_SWEEP, "vllm", "qwen-tts"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vllm",
        model="qwen-tts",
        config_name="tts_vllm_omni.yaml",
        temp_prefix="tts_vllm_omni_sweep",
        run_config_template="tts_vllm_omni_c{concurrency}.yaml",
        run_name_template="tts_vllm_omni_c_{concurrency}_10_minutes",
        default_concurrencies=(1, 2, 4, 8, 16, 32, 64, 128),
    ),
    (CONCURRENCY_SWEEP, "vajra", "qwen3-omni"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vajra",
        model="qwen3-omni",
        config_name="vajra_qwen.yaml",
        temp_prefix="vajra_qwen3_omni_sweep",
        run_config_template="vajra_qwen3_omni_c{concurrency}.yaml",
        run_name_template="vajra_qwen3_omni_c_{concurrency}_10_minutes",
        default_concurrencies=(1, 2, 4, 8, 16, 32, 64, 128),
    ),
    (CONCURRENCY_SWEEP, "vllm", "qwen3-omni"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vllm",
        model="qwen3-omni",
        config_name="qwen3_omni.yaml",
        temp_prefix="vllm_qwen3_omni_sweep",
        run_config_template="vllm_qwen3_omni_c{concurrency}.yaml",
        run_name_template="vllm_qwen3_omni_c_{concurrency}_10_minutes",
        default_concurrencies=(1, 2, 4, 8, 16, 32, 64, 128),
    ),
    (CONCURRENCY_SWEEP, "voxserve", "qwen-tts"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="voxserve",
        model="qwen-tts",
        config_name="tts_voxserve.yaml",
        temp_prefix="tts_voxserve_sweep",
        run_config_template="tts_voxserve_c{concurrency}.yaml",
        run_name_template="voxserve_qwen3_tts_{date_tag}_c={concurrency}",
        default_concurrencies=(1,),
    ),
    (CONCURRENCY_SWEEP, "vajra", "vibe-voice"): SweepSpec(
        sweep_type=CONCURRENCY_SWEEP,
        engine="vajra",
        model="vibe-voice",
        config_name="vajra_vibe_voice.yaml",
        temp_prefix="vajra_vibe_voice_0_5_sweep",
        run_config_template="vajra_vibe_voice_c{concurrency}.yaml",
        run_name_template="vj_vibe_voice_0.5_{date_tag}_c={concurrency}",
        default_concurrencies=(1, 2, 4, 8),
        write_runtime_limits=False,
    ),
    (INPUT_SWEEP, "vajra", "qwen-tts"): SweepSpec(
        sweep_type=INPUT_SWEEP,
        engine="vajra",
        model="qwen-tts",
        config_name="vajra.yaml",
        temp_prefix="vajra_qwen_tts_inputsweep",
        run_config_template="vajra_qwen_c{concurrency}_chars{input_size}.yaml",
        run_name_template=(
            "vajra_qwen3tts_c_{concurrency}_chars_{input_size}_10_minutes"
        ),
        default_concurrency=64,
        default_range_start=380,
        default_range_end=500,
        default_step=40,
        disable_audio_for_input=True,
    ),
    (INPUT_SWEEP, "vllm", "qwen-tts"): SweepSpec(
        sweep_type=INPUT_SWEEP,
        engine="vllm",
        model="qwen-tts",
        config_name="tts_vllm_omni.yaml",
        temp_prefix="vllm_qwen_tts_inputsweep",
        run_config_template="vllm_qwen_tts_c{concurrency}_chars{input_size}.yaml",
        run_name_template=(
            "vllm_qwen3tts_c_{concurrency}_chars_{input_size}_10_minutes"
        ),
        default_concurrency=16,
        default_range_start=180,
        default_range_end=500,
        default_step=40,
        disable_audio_for_input=True,
    ),
    (INPUT_SWEEP, "vllm", "qwen3-omni"): SweepSpec(
        sweep_type=INPUT_SWEEP,
        engine="vllm",
        model="qwen3-omni",
        config_name="qwen3_omni.yaml",
        temp_prefix="vllm_qwen3_omni_inputsweep",
        run_config_template="vllm_qwen3_omni_c{concurrency}_chars{input_size}.yaml",
        run_name_template=(
            "vllm_qwen3_omni_c_{concurrency}_chars_{input_size}_10_minutes"
        ),
        default_concurrency=16,
        default_range_start=20,
        default_range_end=500,
        default_step=40,
        disable_audio_for_input=True,
    ),
}


def _parse_csv_ints(raw: str, *, name: str) -> Tuple[int, ...]:
    values: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{name} must be comma-separated integers"
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{name} values must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return tuple(values)


def _parse_concurrencies(raw: str) -> Tuple[int, ...]:
    return _parse_csv_ints(raw, name="concurrencies")


def _parse_sizes(raw: str) -> Tuple[int, ...]:
    return _parse_csv_ints(raw, name="sizes")


def _parse_trace(raw: str) -> str:
    key = raw.strip().lower().replace("-", "_")
    if key not in TRACE_ALIASES:
        supported = ", ".join(sorted(TRACE_ALIASES))
        raise argparse.ArgumentTypeError(f"trace must be one of: {supported}")
    return TRACE_ALIASES[key]


def _supported_combinations() -> str:
    rows = sorted(SPECS)
    return "\n".join(
        f"  {kind:11s} {engine:8s} {model}" for kind, engine, model in rows
    )


def _normalize_model(raw: str) -> str:
    key = raw.strip().lower()
    if key not in MODEL_ALIASES:
        raise SystemExit(
            f"Unsupported model: {raw}\nSupported combinations:\n{_supported_combinations()}"
        )
    return MODEL_ALIASES[key]


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Base config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Base config must be a YAML mapping: {path}")
    return data


def _write_config(path: Path, config: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, width=1_000_000)


def _mapping_at(config: Dict[str, Any], path: Sequence[str]) -> Dict[str, Any]:
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


def _set_required(config: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    parent = _mapping_at(config, path[:-1])
    key = path[-1]
    if key not in parent:
        raise KeyError(f"Missing YAML key: {'.'.join(path)}")
    parent[key] = value


def _set_mapping_value(config: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    parent = _mapping_at(config, path[:-1])
    parent[path[-1]] = value


def _disable_audio_saving(config: Dict[str, Any]) -> None:
    client = config.get("client")
    if isinstance(client, dict) and "save_audio" in client:
        client["save_audio"] = False

    evaluators = config.get("evaluators")
    if not isinstance(evaluators, list):
        return
    for evaluator in evaluators:
        if not isinstance(evaluator, dict):
            continue
        audio_channel = evaluator.get("audio_channel")
        if isinstance(audio_channel, dict) and "save_audio_files" in audio_channel:
            audio_channel["save_audio_files"] = False


def _apply_trace_source(config: Dict[str, Any], args: argparse.Namespace) -> None:
    session_generator = _mapping_at(config, ("session_generator",))
    if session_generator.get("type") != "trace":
        raise ValueError(
            "--trace can only be used with trace session generator configs"
        )

    flavor = session_generator.get("flavor")
    if not isinstance(flavor, dict):
        raise TypeError("Expected YAML mapping at session_generator.flavor")

    if args.trace == TRACE_SHAREGPT:
        flavor["type"] = "sharegpt"
        if "assistant_role" not in flavor:
            flavor["assistant_role"] = "gpt"
        return

    if args.trace != TRACE_SEED_TTS:
        raise ValueError(f"Unsupported trace source: {args.trace}")

    seed_flavor = {"type": "seed_tts_text"}

    for key in ("min_tokens", "max_tokens", "min_chars", "max_chars"):
        if key in flavor:
            seed_flavor[key] = flavor[key]

    session_generator["trace_file"] = ""
    session_generator["flavor"] = seed_flavor


def _date_tag() -> str:
    return datetime.now().strftime("%m_%d")


def _format_template(
    template: str, *, concurrency: int, input_size: Optional[int] = None
) -> str:
    return template.format(
        concurrency=concurrency,
        input_size=input_size,
        date_tag=_date_tag(),
    )


def _input_sizes(args: argparse.Namespace, spec: SweepSpec) -> Tuple[int, ...]:
    if args.sizes:
        return args.sizes

    start = (
        args.range_start if args.range_start is not None else spec.default_range_start
    )
    end = args.range_end if args.range_end is not None else spec.default_range_end
    step = args.step if args.step is not None else spec.default_step
    if start is None or end is None or step is None:
        raise ValueError("Input sweep requires --sizes or range defaults")
    if start <= 0 or end <= 0 or step <= 0:
        raise ValueError("--range-start, --range-end, and --step must be positive")
    if start > end:
        raise ValueError("--range-start must be <= --range-end")
    return tuple(range(start, end + 1, step))


def _build_run_config(
    base_config: Dict[str, Any],
    spec: SweepSpec,
    *,
    concurrency: int,
    input_size: Optional[int],
    timeout_seconds: int,
    max_sessions: int,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)

    _set_required(
        config,
        ("traffic_scheduler", "target_concurrent_sessions"),
        concurrency,
    )
    _set_required(config, ("runtime", "num_client_threads"), concurrency)
    if spec.write_runtime_limits:
        _set_required(config, ("runtime", "benchmark_timeout"), timeout_seconds)
        _set_required(config, ("runtime", "max_sessions"), max_sessions)

    run_name = _format_template(
        spec.run_name_template,
        concurrency=concurrency,
        input_size=input_size,
    )
    _set_required(config, ("wandb", "run_name"), run_name)

    if spec.sweep_type == INPUT_SWEEP:
        if input_size is None:
            raise ValueError("input_size is required for input sweeps")
        _set_mapping_value(
            config, ("session_generator", "flavor", "min_chars"), input_size
        )
        _set_mapping_value(
            config, ("session_generator", "flavor", "max_chars"), input_size
        )
        if spec.disable_audio_for_input:
            _disable_audio_saving(config)

    return config


def _benchmark_command(config_path: Path) -> List[str]:
    return [
        "python",
        "-Xgil=0",
        "-m",
        "veeksha.benchmark",
        "--benchmark-config-from-file",
        str(config_path),
    ]


def _print_run_header(
    *,
    spec: SweepSpec,
    run_index: int,
    run_count: int,
    concurrency: int,
    input_size: Optional[int],
    run_config: Path,
    run_name: str,
    timeout_seconds: int,
    max_sessions: int,
    dry_run: bool,
    trace_source: str,
) -> None:
    mode = "DRY RUN " if dry_run else ""
    target = f"concurrency={concurrency}"
    if input_size is not None:
        target += f" input_chars={input_size}"
    print("=" * 62)
    print(
        f" {mode}Running {spec.engine}/{spec.model} {spec.sweep_type} sweep: "
        f"{target} ({run_index}/{run_count})"
    )
    print(f"   config       : {run_config}")
    print(f"   base config  : {spec.config_path}")
    print(f"   trace        : {trace_source}")
    print(f"   wandb        : {run_name}")
    if spec.write_runtime_limits:
        print(f"   timeout      : {timeout_seconds}s")
        print(f"   max_sessions : {max_sessions}")
    print("=" * 62)


def _run_sweep(args: argparse.Namespace, spec: SweepSpec) -> int:
    base_config = _load_config(spec.config_path)
    _apply_trace_source(base_config, args)
    tmp_parent = Path(tempfile.mkdtemp(prefix=f"{spec.temp_prefix}."))

    if spec.sweep_type == CONCURRENCY_SWEEP:
        concurrencies = args.concurrencies or spec.default_concurrencies
        work_items = [(concurrency, None) for concurrency in concurrencies]
    else:
        concurrency = args.concurrency or spec.default_concurrency
        if concurrency is None:
            raise ValueError("Input sweep requires --concurrency")
        work_items = [(concurrency, size) for size in _input_sizes(args, spec)]

    run_count = len(work_items)
    try:
        for index, (concurrency, input_size) in enumerate(work_items, start=1):
            config_name = _format_template(
                spec.run_config_template,
                concurrency=concurrency,
                input_size=input_size,
            )
            run_config = tmp_parent / config_name
            run_name = _format_template(
                spec.run_name_template,
                concurrency=concurrency,
                input_size=input_size,
            )
            run_cfg = _build_run_config(
                base_config,
                spec,
                concurrency=concurrency,
                input_size=input_size,
                timeout_seconds=args.timeout_seconds,
                max_sessions=args.max_sessions,
            )
            _write_config(run_config, run_cfg)
            _print_run_header(
                spec=spec,
                run_index=index,
                run_count=run_count,
                concurrency=concurrency,
                input_size=input_size,
                run_config=run_config,
                run_name=run_name,
                timeout_seconds=args.timeout_seconds,
                max_sessions=args.max_sessions,
                dry_run=args.dry_run,
                trace_source=args.trace,
            )
            command = _benchmark_command(run_config)
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, cwd=REPO_ROOT, check=True)
                if index < run_count and args.cooldown_seconds > 0:
                    print(f"-- cooldown {args.cooldown_seconds}s before next run --")
                    time.sleep(args.cooldown_seconds)
    finally:
        if args.dry_run:
            print(f"Dry-run configs left in: {tmp_parent}")
        else:
            for child in tmp_parent.iterdir():
                child.unlink()
            tmp_parent.rmdir()

    if spec.sweep_type == CONCURRENCY_SWEEP:
        values = " ".join(str(c) for c, _ in work_items)
        print(f"Sweep complete for concurrencies: {values}")
    else:
        sizes = " ".join(str(size) for _, size in work_items if size is not None)
        concurrency = work_items[0][0] if work_items else "-"
        print(f"Input-size sweep complete (concurrency={concurrency}, sizes={sizes})")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sweep-type", choices=(CONCURRENCY_SWEEP, INPUT_SWEEP), required=True
    )
    parser.add_argument(
        "--engine", choices=("vajra", "vllm", "voxserve"), required=True
    )
    parser.add_argument(
        "--model",
        choices=tuple(sorted(MODEL_ALIASES)),
        required=True,
        help="Model/workload alias. qwen3-tts and tts map to qwen-tts.",
    )
    parser.add_argument(
        "--trace",
        type=_parse_trace,
        default=TRACE_SHAREGPT,
        help="Trace text source. Supported: sharegpt, seed_tts.",
    )
    parser.add_argument(
        "--concurrencies",
        type=_parse_concurrencies,
        help="Comma-separated concurrency values for concurrency sweeps.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Fixed concurrency for input sweeps.",
    )
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        help="Comma-separated input sizes for input sweeps.",
    )
    parser.add_argument("--range-start", type=int, help="Input sweep range start.")
    parser.add_argument("--range-end", type=int, help="Input sweep range end.")
    parser.add_argument("--step", type=int, help="Input sweep range step.")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--cooldown-seconds", type=int, default=120)
    parser.add_argument("--max-sessions", type=int, default=20000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write temporary configs and print commands without running benchmarks.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be non-negative")
    if args.max_sessions <= 0:
        parser.error("--max-sessions must be positive")
    if args.concurrency is not None and args.concurrency <= 0:
        parser.error("--concurrency must be positive")

    model = _normalize_model(args.model)
    spec = SPECS.get((args.sweep_type, args.engine, model))
    if spec is None:
        parser.error(
            "Unsupported sweep combination:\n"
            f"  {args.sweep_type} {args.engine} {model}\n"
            f"Supported combinations:\n{_supported_combinations()}"
        )

    if args.sweep_type == CONCURRENCY_SWEEP:
        disallowed = [
            name
            for name in (
                "--concurrency",
                "--sizes",
                "--range-start",
                "--range-end",
                "--step",
            )
            if getattr(args, name.lstrip("-").replace("-", "_")) is not None
        ]
        if disallowed:
            parser.error(
                "concurrency sweeps do not accept input-sweep options: "
                + ", ".join(disallowed)
            )
    elif args.concurrencies is not None:
        parser.error("input sweeps use --concurrency, not --concurrencies")

    return _run_sweep(args, spec)


if __name__ == "__main__":
    sys.exit(main())
