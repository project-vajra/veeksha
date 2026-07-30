"""CLI runner for Veeksha benchmark invocations.

We keep the CLI entrypoint intentionally thin and centralize orchestration
logic here for readability and reuse.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone

from veeksha.benchmark import (
    manage_benchmark_run,
    run_benchmark_with_endpoint,
)
from veeksha.config.benchmark import BenchmarkConfig, carry_sidecar_attrs
from veeksha.config.utils import to_serializable_config_dict
from veeksha.logger import init_logger
from veeksha.named_benchmark.resolve import NamedBenchmarkError, resolve_named_benchmark
from veeksha.orchestration.benchmark_orchestrator import managed_server
from veeksha.sweep_summary import write_sweep_summary
from veeksha.wandb_integration import dedup_tags, maybe_log_sweep_summary

logger = init_logger(__name__)


def _maybe_resolve_named(cfg: BenchmarkConfig) -> BenchmarkConfig:
    """If ``cfg.benchmark`` is set, fetch the definition and apply free variables."""
    if not cfg.benchmark:
        return cfg
    knob_overrides = getattr(cfg, "_knob_overrides", None)
    try:
        resolved, meta = resolve_named_benchmark(cfg, knob_overrides=knob_overrides)
    except NamedBenchmarkError as exc:
        raise SystemExit(str(exc)) from exc
    # Carry CLI-provided keys and named-benchmark meta onto the resolved config
    # (frozen dataclasses only accept new attrs via object.__setattr__).
    provided = getattr(cfg, "_cli_provided_keys", None)
    if provided is not None:
        object.__setattr__(resolved, "_cli_provided_keys", provided)
    if knob_overrides is not None:
        object.__setattr__(resolved, "_knob_overrides", knob_overrides)
    object.__setattr__(resolved, "_named_benchmark_meta", meta)
    return resolved


class BenchmarkCliRunner:
    """Runs one or more `BenchmarkConfig`s produced by the CLI/YAML loader."""

    def __init__(self, benchmark_configs: list[BenchmarkConfig]):
        self._benchmark_configs = [_maybe_resolve_named(c) for c in benchmark_configs]

    @classmethod
    def from_cli(cls) -> "BenchmarkCliRunner":
        import sys

        from veeksha.cli.free_variables import parse_benchmark_run_configs

        return cls(parse_benchmark_run_configs(sys.argv[1:]))

    def run_all(self) -> None:
        """Run all benchmark configs, grouping sweeps by base output directory."""
        configs_by_base_dir: dict[str, list[BenchmarkConfig]] = defaultdict(list)
        for cfg in self._benchmark_configs:
            configs_by_base_dir[cfg.output_dir].append(cfg)

        for base_output_dir, configs in configs_by_base_dir.items():
            if len(configs) > 1:
                self._run_sweep(base_output_dir=base_output_dir, configs=configs)
            else:
                self._run_single(configs[0])

    def _run_single(self, cfg: BenchmarkConfig) -> None:
        manage_benchmark_run(cfg)

    def _run_sweep(
        self, *, base_output_dir: str, configs: list[BenchmarkConfig]
    ) -> None:
        sweep_timestamp = datetime.now(timezone.utc).strftime("%d:%m:%Y-%H:%M:%S")
        sweep_dir = os.path.join(base_output_dir, f"sweep_{sweep_timestamp}")
        os.makedirs(sweep_dir, exist_ok=True)
        sweep_group = f"sweep-{os.path.basename(sweep_dir.rstrip('/'))}"

        run_dirs_by_config_idx: dict[int, str] = {}
        num_managed_groups = 0

        for server_group in _group_configs_by_server(configs):
            server = server_group[0][1].server
            if server is None:
                for config_idx, cfg in server_group:
                    updated_cfg = _prepare_sweep_config(
                        cfg,
                        sweep_dir=sweep_dir,
                        sweep_group=sweep_group,
                    )
                    manage_benchmark_run(updated_cfg)
                    run_dirs_by_config_idx[config_idx] = updated_cfg.output_dir
                continue

            num_managed_groups += 1
            logger.info(
                "Running %d sweep configs with one %s server lifecycle",
                len(server_group),
                server.engine,
            )
            server_output_dir = os.path.join(
                sweep_dir,
                f"managed_server_{num_managed_groups:02d}",
            )
            with managed_server(server, output_dir=server_output_dir) as server_info:
                endpoint = server_info["endpoint"]
                for config_idx, cfg in server_group:
                    updated_cfg = _prepare_sweep_config(
                        cfg,
                        sweep_dir=sweep_dir,
                        sweep_group=sweep_group,
                    )
                    run_benchmark_with_endpoint(updated_cfg, endpoint)
                    run_dirs_by_config_idx[config_idx] = updated_cfg.output_dir

        all_run_dirs = [
            run_dirs_by_config_idx[config_idx] for config_idx in range(len(configs))
        ]

        write_sweep_summary(sweep_dir, all_run_dirs)

        first = configs[0]
        if first.wandb.enabled:
            maybe_log_sweep_summary(
                sweep_dir=sweep_dir,
                wandb_cfg=first.wandb,
                group=first.wandb.group or sweep_group,
            )


def _prepare_sweep_config(
    cfg: BenchmarkConfig,
    *,
    sweep_dir: str,
    sweep_group: str,
) -> BenchmarkConfig:
    wandb_cfg = cfg.wandb
    if wandb_cfg.enabled:
        wandb_cfg = replace(
            wandb_cfg,
            group=wandb_cfg.group or sweep_group,
            tags=dedup_tags([*wandb_cfg.tags, "sweep"]),
        )
    return carry_sidecar_attrs(cfg, replace(cfg, output_dir=sweep_dir, wandb=wandb_cfg))


def _group_configs_by_server(
    configs: list[BenchmarkConfig],
) -> list[list[tuple[int, BenchmarkConfig]]]:
    configs_by_server: dict[
        str | None,
        list[tuple[int, BenchmarkConfig]],
    ] = {}
    for config_idx, cfg in enumerate(configs):
        server_key = _server_config_key(cfg)
        configs_by_server.setdefault(server_key, []).append((config_idx, cfg))
    return list(configs_by_server.values())


def _server_config_key(cfg: BenchmarkConfig) -> str | None:
    if cfg.server is None:
        return None
    return json.dumps(
        to_serializable_config_dict(cfg.server),
        sort_keys=True,
        separators=(",", ":"),
    )


def run_cli(configs: list[BenchmarkConfig]) -> None:
    """Run benchmark configs parsed from CLI."""
    BenchmarkCliRunner(configs).run_all()


def main() -> None:
    import sys

    from veeksha.cli.free_variables import parse_benchmark_run_configs

    run_cli(parse_benchmark_run_configs(sys.argv[1:]))
