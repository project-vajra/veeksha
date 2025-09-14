import multiprocessing
import platform

from veeksha.config.microbenchmark import MicrobenchmarkConfig
from veeksha.microbenchmark import Microbenchmark


def run():
    configs = MicrobenchmarkConfig.create_from_cli_args()
    for config in configs:
        microbenchmark = Microbenchmark(config)
        microbenchmark.run()


def main():
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    run()


if __name__ == "__main__":
    main()
