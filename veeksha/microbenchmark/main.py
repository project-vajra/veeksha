import multiprocessing
import platform

from veeksha.config.microbenchmark import MicrobenchmarkConfig
from veeksha.microbenchmark.microbenchmark import Microbenchmark

if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    configs = MicrobenchmarkConfig.create_from_cli_args()
    for config in configs:
        config.write_config_to_file()
        microbenchmark = Microbenchmark(config)
        microbenchmark.run()
