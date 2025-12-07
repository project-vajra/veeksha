from veeksha.new.benchmark import run_benchmark
from veeksha.new.config.benchmark import BenchmarkConfig


def main():
    # instantiate benchmark configuration from CLI
    # run benchmark
    benchmark_config = BenchmarkConfig.create_from_cli_args()
    run_benchmark(benchmark_config)


if __name__ == "__main__":
    main()
