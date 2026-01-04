from veeksha.new.benchmark import manage_benchmark_run
from veeksha.new.config.benchmark import BenchmarkConfig


def main():
    benchmark_configs = BenchmarkConfig.create_from_cli_args()
    for benchmark_config in benchmark_configs:
        manage_benchmark_run(benchmark_config)


if __name__ == "__main__":
    main()
