import os

from veeksha.config.microbenchmark import MicrobenchmarkConfig
from veeksha.logger import init_logger
from veeksha.microbenchmark.decode_profiler import DecodeProfiler
from veeksha.microbenchmark.prefill_profiler import PrefillProfiler

# pyright: reportCallIssue=false, reportArgumentType=false
logger = init_logger(__name__)


class Microbenchmark:
    def __init__(
        self,
        microbenchmark_config: MicrobenchmarkConfig,
    ) -> None:
        self.microbenchmark_config = microbenchmark_config

        # Store base benchmark config
        self.base_benchmark_config = microbenchmark_config.create_benchmark_config()

    def _run_prefill_profiler(self) -> None:
        """Run the prefill profiler."""
        logger.info("Running prefill profiler...")

        # Create prefill-specific output directory
        prefill_output_dir = os.path.join(
            self.microbenchmark_config.output_dir, "prefill"
        )

        # Create benchmark config with prefill output directory
        benchmark_config_with_prefill = (
            self.microbenchmark_config.create_benchmark_config(
                output_dir=prefill_output_dir
            )
        )

        # Run prefill profiler
        prefill_profiler = PrefillProfiler(
            benchmark_config_with_prefill,
            self.microbenchmark_config.prefill_profiler.prefill_lengths,
        )
        prefill_profiler.run()
        logger.info("Prefill profiler completed.")

    def _run_decode_profiler(self) -> None:
        """Run the decode profiler."""
        logger.info("Running decode profiler...")

        # Create decode-specific output directory
        decode_output_dir = os.path.join(
            self.microbenchmark_config.output_dir, "decode"
        )

        # Create benchmark config with decode output directory
        benchmark_config_with_decode = (
            self.microbenchmark_config.create_benchmark_config(
                output_dir=decode_output_dir
            )
        )

        # Run decode profiler
        decode_profiler = DecodeProfiler(
            benchmark_config_with_decode, self.microbenchmark_config.decode_profiler
        )
        decode_profiler.run()
        logger.info("Decode profiler completed.")

    def run(self):
        """Run all enabled microbenchmark profilers."""
        logger.info(
            f"Starting microbenchmark profiling with output dir: {self.microbenchmark_config.output_dir}"
        )

        if self.microbenchmark_config.prefill_profiler.enabled:
            self._run_prefill_profiler()

        if self.microbenchmark_config.decode_profiler.enabled:
            self._run_decode_profiler()

        if (
            not self.microbenchmark_config.prefill_profiler.enabled
            and not self.microbenchmark_config.decode_profiler.enabled
        ):
            logger.warning(
                "No profilers enabled. Enable prefill_profiler or decode_profiler in config."
            )

        logger.info("Microbenchmark profiling completed!")


if __name__ == "__main__":
    configs = MicrobenchmarkConfig.create_from_cli_args()
    for config in configs:
        config.write_config_to_file()
        profiler = Microbenchmark(config)
        profiler.run()
