import os
from dataclasses import replace

from veeksha.config.microbenchmarks import MicrobenchmarkConfig
from veeksha.logger import init_logger
from veeksha.microbenchmarks.prefill_profiler import PrefillProfiler
from veeksha.microbenchmarks.decode_profiler import DecodeProfiler

# pyright: reportCallIssue=false, reportArgumentType=false
logger = init_logger(__name__)


class Microbenchmark:
    def __init__(
        self,
        microbenchmark_config: MicrobenchmarkConfig,
    ) -> None:
        self.microbenchmark_config = microbenchmark_config

        # Update benchmark config with the microbenchmark output directory
        self.benchmark_config = replace(
            microbenchmark_config.benchmark_config,
            metrics_config=replace(
                microbenchmark_config.benchmark_config.metrics_config,
                output_dir=microbenchmark_config.output_dir
            )
        )

    def _run_prefill_profiler(self) -> None:
        """Run the prefill profiler."""
        logger.info("Running prefill profiler...")
        
        # Create prefill-specific output directory
        prefill_output_dir = os.path.join(self.microbenchmark_config.output_dir, "prefill")
        
        # Create benchmark config with prefill output directory
        benchmark_config_with_prefill = replace(
            self.benchmark_config,
            metrics_config=replace(
                self.benchmark_config.metrics_config,
                output_dir=prefill_output_dir
            )
        )
        
        # Create prefill profiler config
        prefill_config = self.microbenchmark_config.create_prefill_profiler_config()
        
        # Run prefill profiler
        prefill_profiler = PrefillProfiler(benchmark_config_with_prefill, prefill_config)
        prefill_profiler.run()
        logger.info("Prefill profiler completed.")

    def _run_decode_profiler(self) -> None:
        """Run the decode profiler."""
        logger.info("Running decode profiler...")
        
        # Create decode-specific output directory
        decode_output_dir = os.path.join(self.microbenchmark_config.output_dir, "decode")
        
        # Create benchmark config with decode output directory
        benchmark_config_with_decode = replace(
            self.benchmark_config,
            metrics_config=replace(
                self.benchmark_config.metrics_config,
                output_dir=decode_output_dir
            )
        )
        
        # Create decode profiler config
        decode_config = self.microbenchmark_config.create_decode_profiler_config()
        
        # Run decode profiler
        decode_profiler = DecodeProfiler(benchmark_config_with_decode, decode_config)
        decode_profiler.run()
        logger.info("Decode profiler completed.")

    def run(self):
        """Run all enabled microbenchmark profilers."""
        logger.info(f"Starting microbenchmark profiling with output dir: {self.microbenchmark_config.output_dir}")
        
        if self.microbenchmark_config.prefill_profiler.enabled:
            self._run_prefill_profiler()
        
        if self.microbenchmark_config.decode_profiler.enabled:
            self._run_decode_profiler()
        
        if not self.microbenchmark_config.prefill_profiler.enabled and not self.microbenchmark_config.decode_profiler.enabled:
            logger.warning("No profilers enabled. Enable prefill_profiler or decode_profiler in config.")
        
        logger.info("Microbenchmark profiling completed!")


if __name__ == "__main__":
    configs = MicrobenchmarkConfig.create_from_cli_args()
    for config in configs:
        config.write_config_to_file()
        profiler = Microbenchmark(config)
        profiler.run()