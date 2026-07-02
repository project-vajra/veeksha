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

    # Override output_dir for generated benchmark configs
    python scripts/sweep.py --sweep-type concurrency --engine vajra \\
        --model qwen-tts --output-dir 'benchmark_output/{run_name}'
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

from veeksha.config.endpoint import EndpointConfig
from veeksha.sweeps.config import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_MAX_SESSIONS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DEFAULT_TIMEOUT_SECONDS,
    TRACE_ALIASES,
    TRACE_SHAREGPT,
    SweepConfig,
    SweepConfigError,
)
from veeksha.sweeps.specs import (
    CONCURRENCY_SWEEP,
    INPUT_SWEEP,
    MODEL_ALIASES,
    REPO_ROOT,
    SPECS,
    SweepSpec,
    supported_combinations,
)
from veeksha.sweeps.utils import (
    apply_endpoint,
    apply_trace_source,
    benchmark_command,
    build_run_config,
    format_template,
    input_sizes,
    load_config,
    write_config,
)


@dataclass(frozen=True)
class SweepRunDescriptor:
    run_index: int
    run_count: int
    concurrency: int
    input_size: Optional[int]
    run_config: Path
    run_name: str
    output_dir: Optional[str]
    command: List[str]
    timeout_seconds: int
    max_sessions: int
    trace_source: str


@dataclass(frozen=True)
class SweepPlan:
    spec: SweepSpec
    tmp_parent: Path
    runs: Tuple[SweepRunDescriptor, ...]
    base_config_path: Path
    endpoint: Optional[EndpointConfig] = None

    def cleanup(self) -> None:
        for child in self.tmp_parent.iterdir():
            child.unlink()
        self.tmp_parent.rmdir()


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


def _normalize_model(raw: str) -> str:
    key = raw.strip().lower()
    if key not in MODEL_ALIASES:
        raise SweepConfigError(
            f"Unsupported model: {raw}\nSupported combinations:\n{supported_combinations()}"
        )
    return MODEL_ALIASES[key]


def _resolve_base_config_path(sweep_config: SweepConfig, spec: SweepSpec) -> Path:
    if sweep_config.base_config is None:
        return spec.config_path
    path = Path(sweep_config.base_config)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise SweepConfigError(f"sweep.base_config does not exist: {path}")
    return path

def _build_sweep_plan(
    sweep_config: SweepConfig,
    spec: SweepSpec,
    *,
    endpoint: Optional[EndpointConfig] = None,
    tmp_parent: Optional[Path] = None,
) -> SweepPlan:
    base_config_path = _resolve_base_config_path(sweep_config, spec)
    base_config = load_config(base_config_path)
    apply_trace_source(base_config, sweep_config)
    if tmp_parent is None:
        tmp_parent = Path(tempfile.mkdtemp(prefix=f"{spec.temp_prefix}."))
    else:
        tmp_parent.mkdir(parents=True, exist_ok=True)

    if spec.sweep_type == CONCURRENCY_SWEEP:
        concurrencies = sweep_config.concurrencies or spec.default_concurrencies
        work_items = [(concurrency, None) for concurrency in concurrencies]
    else:
        concurrency = sweep_config.concurrency or spec.default_concurrency
        if concurrency is None:
            raise ValueError("Input sweep requires --concurrency")
        work_items = [(concurrency, size) for size in input_sizes(sweep_config, spec)]

    runs: List[SweepRunDescriptor] = []
    run_count = len(work_items)
    for index, (concurrency, input_size) in enumerate(work_items, start=1):
        config_name = format_template(
            spec.run_config_template,
            concurrency=concurrency,
            input_size=input_size,
        )
        run_config = tmp_parent / config_name
        run_name = format_template(
            spec.run_name_template,
            concurrency=concurrency,
            input_size=input_size,
        )
        run_cfg = build_run_config(
            base_config,
            spec,
            concurrency=concurrency,
            input_size=input_size,
            timeout_seconds=sweep_config.timeout_seconds,
            max_sessions=sweep_config.max_sessions,
            output_dir_template=sweep_config.output_dir,
            min_tokens=sweep_config.min_tokens,
            max_tokens=sweep_config.max_tokens,
            min_chars=sweep_config.min_chars,
            max_chars=sweep_config.max_chars,
        )
        apply_endpoint(run_cfg, endpoint)
        write_config(run_config, run_cfg)
        runs.append(
            SweepRunDescriptor(
                run_index=index,
                run_count=run_count,
                concurrency=concurrency,
                input_size=input_size,
                run_config=run_config,
                run_name=run_name,
                output_dir=run_cfg.get("output_dir"),
                command=benchmark_command(run_config),
                timeout_seconds=sweep_config.timeout_seconds,
                max_sessions=sweep_config.max_sessions,
                trace_source=sweep_config.trace,
            )
        )

    return SweepPlan(
        spec=spec,
        tmp_parent=tmp_parent,
        runs=tuple(runs),
        base_config_path=base_config_path,
        endpoint=endpoint,
    )


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
    output_dir: Optional[str],
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
    if output_dir is not None:
        print(f"   output_dir   : {output_dir}")
    print(f"   wandb        : {run_name}")
    if spec.write_runtime_limits:
        print(f"   timeout      : {timeout_seconds}s")
        print(f"   max_sessions : {max_sessions}")
    print("=" * 62)


def _run_sweep(sweep_config: SweepConfig, spec: SweepSpec, *, dry_run: bool) -> int:
    plan = _build_sweep_plan(sweep_config, spec)
    try:
        for run in plan.runs:
            _print_run_header(
                spec=spec,
                run_index=run.run_index,
                run_count=run.run_count,
                concurrency=run.concurrency,
                input_size=run.input_size,
                run_config=run.run_config,
                run_name=run.run_name,
                timeout_seconds=run.timeout_seconds,
                max_sessions=run.max_sessions,
                dry_run=dry_run,
                trace_source=run.trace_source,
                output_dir=run.output_dir,
            )
            print(" ".join(run.command))
            if not dry_run:
                subprocess.run(run.command, cwd=REPO_ROOT, check=True)
                if run.run_index < run.run_count and sweep_config.cooldown_seconds > 0:
                    print(
                        f"-- cooldown {sweep_config.cooldown_seconds}s before next run --"
                    )
                    time.sleep(sweep_config.cooldown_seconds)
    finally:
        if dry_run:
            print(f"Dry-run configs left in: {plan.tmp_parent}")
        else:
            plan.cleanup()

    if spec.sweep_type == CONCURRENCY_SWEEP:
        values = " ".join(str(run.concurrency) for run in plan.runs)
        print(f"Sweep complete for concurrencies: {values}")
    else:
        sizes = " ".join(
            str(run.input_size) for run in plan.runs if run.input_size is not None
        )
        concurrency = plan.runs[0].concurrency if plan.runs else "-"
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
    parser.add_argument("--engine", choices=("vajra", "vllm", "sglang"), required=True)
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
        "--output-dir",
        help=(
            "Override benchmark output_dir in generated configs. Supports "
            "{concurrency}, {input_size}, {run_name}, {date_tag}, {engine}, "
            "{model}, and {sweep_type}."
        ),
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        help="Minimum input word count for concurrency sweeps. Defaults to 20.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum input word count for concurrency sweeps. Defaults to 150.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        help="Minimum input char count for concurrency sweeps.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        help="Maximum input char count for concurrency sweeps.",
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
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--cooldown-seconds", type=int, default=DEFAULT_COOLDOWN_SECONDS
    )
    parser.add_argument("--max-sessions", type=int, default=DEFAULT_MAX_SESSIONS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write temporary configs and print commands without running benchmarks.",
    )
    return parser


def _sweep_config_from_args(args: argparse.Namespace) -> SweepConfig:
    return SweepConfig(
        sweep_type=args.sweep_type,
        engine=args.engine,
        model=args.model,
        trace=args.trace,
        output_dir=args.output_dir,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        concurrencies=args.concurrencies,
        concurrency=args.concurrency,
        sizes=args.sizes,
        range_start=args.range_start,
        range_end=args.range_end,
        step=args.step,
        timeout_seconds=args.timeout_seconds,
        cooldown_seconds=args.cooldown_seconds,
        max_sessions=args.max_sessions,
    )


def _validate_length_config(sweep_config: SweepConfig) -> SweepConfig:
    char_pair = (sweep_config.min_chars is not None, sweep_config.max_chars is not None)
    token_pair = (
        sweep_config.min_tokens is not None,
        sweep_config.max_tokens is not None,
    )
    has_chars = any(char_pair)
    has_tokens = any(token_pair)

    if sweep_config.sweep_type == INPUT_SWEEP:
        if has_chars or has_tokens:
            raise SweepConfigError(
                "input sweeps derive min_chars/max_chars from sizes; do not pass "
                "min_tokens/max_tokens or min_chars/max_chars"
            )
        return sweep_config

    if has_chars and has_tokens:
        raise SweepConfigError(
            "min_chars/max_chars and min_tokens/max_tokens are mutually exclusive"
        )
    if has_chars and not all(char_pair):
        raise SweepConfigError("min_chars and max_chars must be specified together")
    if has_tokens and not all(token_pair):
        raise SweepConfigError("min_tokens and max_tokens must be specified together")

    if has_chars:
        min_chars = sweep_config.min_chars
        max_chars = sweep_config.max_chars
        if min_chars is None or max_chars is None:
            raise SweepConfigError("min_chars and max_chars must be specified together")
        if min_chars <= 0 or max_chars <= 0:
            raise SweepConfigError("min_chars and max_chars must be positive")
        if min_chars > max_chars:
            raise SweepConfigError("min_chars must be <= max_chars")
        return sweep_config

    if has_tokens:
        min_tokens = sweep_config.min_tokens
        max_tokens = sweep_config.max_tokens
        if min_tokens is None or max_tokens is None:
            raise SweepConfigError(
                "min_tokens and max_tokens must be specified together"
            )
        if min_tokens <= 0 or max_tokens <= 0:
            raise SweepConfigError("min_tokens and max_tokens must be positive")
        if min_tokens > max_tokens:
            raise SweepConfigError("min_tokens must be <= max_tokens")
        return sweep_config

    return replace(
        sweep_config,
        min_tokens=DEFAULT_MIN_TOKENS,
        max_tokens=DEFAULT_MAX_TOKENS,
    )


def validate_sweep_config(sweep_config: SweepConfig) -> SweepConfig:
    """Validate and normalize a typed sweep config."""
    if sweep_config.timeout_seconds <= 0:
        raise SweepConfigError("timeout_seconds must be positive")
    if sweep_config.cooldown_seconds < 0:
        raise SweepConfigError("cooldown_seconds must be non-negative")
    if sweep_config.max_sessions <= 0:
        raise SweepConfigError("max_sessions must be positive")
    if sweep_config.concurrency is not None and sweep_config.concurrency <= 0:
        raise SweepConfigError("concurrency must be positive")

    sweep_config = _validate_length_config(sweep_config)

    if sweep_config.sweep_type == CONCURRENCY_SWEEP:
        disallowed = []
        for name, value in (
            ("concurrency", sweep_config.concurrency),
            ("sizes", sweep_config.sizes),
            ("range_start", sweep_config.range_start),
            ("range_end", sweep_config.range_end),
            ("step", sweep_config.step),
        ):
            if value is not None:
                disallowed.append(name)
        if disallowed:
            raise SweepConfigError(
                "concurrency sweeps do not accept input-sweep options: "
                + ", ".join(disallowed)
            )
    elif sweep_config.sweep_type == INPUT_SWEEP:
        if sweep_config.concurrencies is not None:
            raise SweepConfigError("input sweeps use concurrency, not concurrencies")
    else:
        raise SweepConfigError(
            f"sweep_type must be one of: {CONCURRENCY_SWEEP}, {INPUT_SWEEP}"
        )

    return sweep_config


def build_parser() -> argparse.ArgumentParser:
    """Build the sweep CLI parser."""
    return _build_parser()


def validate_sweep_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Validate CLI arguments through the typed sweep config boundary."""
    try:
        validate_sweep_config(_sweep_config_from_args(args))
    except SweepConfigError as exc:
        parser.error(str(exc))


def resolve_sweep_config(config: SweepConfig) -> Tuple[SweepConfig, SweepSpec]:
    """Resolve a typed sweep config to a normalized config and sweep spec."""
    config = validate_sweep_config(config)
    model = _normalize_model(config.model)
    config = replace(config, model=model)
    spec = SPECS.get((config.sweep_type, config.engine, model))
    if spec is None:
        raise SweepConfigError(
            "Unsupported sweep combination:\n"
            f"  {config.sweep_type} {config.engine} {model}\n"
            f"Supported combinations:\n{supported_combinations()}"
        )
    return config, spec


def resolve_sweep_spec(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> SweepSpec:
    """Resolve CLI arguments to a sweep specification."""
    try:
        _, spec = resolve_sweep_config(_sweep_config_from_args(args))
    except SweepConfigError as exc:
        parser.error(str(exc))
    return spec


def build_sweep_plan(
    sweep_config: SweepConfig,
    spec: SweepSpec,
    *,
    endpoint: Optional[EndpointConfig] = None,
    tmp_parent: Optional[Path] = None,
) -> SweepPlan:
    """Build a concrete set of benchmark runs for a resolved sweep spec."""
    return _build_sweep_plan(
        sweep_config, spec, endpoint=endpoint, tmp_parent=tmp_parent
    )


def build_sweep_plan_from_config(
    config: SweepConfig,
    *,
    endpoint: Optional[EndpointConfig] = None,
    tmp_parent: Optional[Path] = None,
) -> SweepPlan:
    """Build a concrete sweep plan from a typed programmatic config."""
    config, spec = resolve_sweep_config(config)
    return build_sweep_plan(config, spec, endpoint=endpoint, tmp_parent=tmp_parent)


def run_sweep(sweep_config: SweepConfig, spec: SweepSpec, *, dry_run: bool) -> int:
    """Execute a resolved sweep plan."""
    return _run_sweep(sweep_config, spec, dry_run=dry_run)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        sweep_config, spec = resolve_sweep_config(_sweep_config_from_args(args))
    except SweepConfigError as exc:
        parser.error(str(exc))
    return run_sweep(sweep_config, spec, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
