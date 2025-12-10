from veeksha.new.benchmark import run_benchmark
from veeksha.new.config.benchmark import BenchmarkConfig


def main():
    # instantiate benchmark configuration from CLI
    # run benchmark
    benchmark_configs = BenchmarkConfig.create_from_cli_args()
    for benchmark_config in benchmark_configs:
        run_benchmark(benchmark_config)


if __name__ == "__main__":
    main()
