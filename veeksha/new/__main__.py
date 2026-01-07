import os
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone

from veeksha.new.benchmark import manage_benchmark_run
from veeksha.new.config.benchmark import BenchmarkConfig
from veeksha.new.sweep_summary import write_sweep_summary


def main():
    """Entrypoint for Veeksha benchmarks. Handles single runs and sweeps."""

    benchmark_configs = BenchmarkConfig.create_from_cli_args()

    # group configs by base output directory
    configs_by_base_dir: dict[str, list[BenchmarkConfig]] = defaultdict(list)
    for cfg in benchmark_configs:
        configs_by_base_dir[cfg.output_dir].append(cfg)

    all_run_dirs: dict[str, list[str]] = defaultdict(list)

    for base_output_dir, configs in configs_by_base_dir.items():
        if len(configs) > 1:
            # sweep mode: create a timestamped sweep directory
            sweep_timestamp = datetime.now(timezone.utc).strftime("%d:%m:%Y-%H:%M:%S")
            sweep_dir = os.path.join(base_output_dir, f"sweep_{sweep_timestamp}")
            os.makedirs(sweep_dir, exist_ok=True)

            # update each config's output_dir to be inside the sweep directory
            for idx, cfg in enumerate(configs):
                updated_cfg = replace(cfg, output_dir=sweep_dir)
                manage_benchmark_run(updated_cfg)
                # manage_benchmark_run mutates output_dir to the resolved run dir
                all_run_dirs[sweep_dir].append(updated_cfg.output_dir)

            write_sweep_summary(sweep_dir, all_run_dirs[sweep_dir])
        else:
            # single run
            cfg = configs[0]
            manage_benchmark_run(cfg)


if __name__ == "__main__":
    main()
