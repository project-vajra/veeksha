"""CLI runner for `python -m veeksha.new`.

We keep the module entrypoint (`veeksha/new/__main__.py`) intentionally thin and
centralize orchestration logic here for readability and reuse.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone

from veeksha.benchmark import manage_benchmark_run
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.logger import init_logger
from veeksha.sweep_summary import write_sweep_summary
from veeksha.wandb_integration import dedup_tags, maybe_log_sweep_summary

logger = init_logger(__name__)


class BenchmarkCliRunner:
    """Runs one or more `BenchmarkConfig`s produced by the CLI/YAML loader."""

    def __init__(self, benchmark_configs: list[BenchmarkConfig]):
        self._benchmark_configs = benchmark_configs

    @classmethod
    def from_cli(cls) -> "BenchmarkCliRunner":
        return cls(BenchmarkConfig.create_from_cli_args())

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

        all_run_dirs: list[str] = []

        for cfg in configs:
            wandb_cfg = cfg.wandb
            if wandb_cfg.enabled:
                wandb_cfg = replace(
                    wandb_cfg,
                    group=wandb_cfg.group or sweep_group,
                    tags=dedup_tags([*wandb_cfg.tags, "sweep"]),
                )

            updated_cfg = replace(cfg, output_dir=sweep_dir, wandb=wandb_cfg)
            manage_benchmark_run(updated_cfg)
            # manage_benchmark_run mutates output_dir to the resolved run dir
            all_run_dirs.append(updated_cfg.output_dir)

        write_sweep_summary(sweep_dir, all_run_dirs)

        first = configs[0]
        if first.wandb.enabled:
            maybe_log_sweep_summary(
                sweep_dir=sweep_dir,
                wandb_cfg=first.wandb,
                group=first.wandb.group or sweep_group,
            )


def main() -> None:
    BenchmarkCliRunner.from_cli().run_all()
