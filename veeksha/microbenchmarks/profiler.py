import multiprocessing
import os
import platform

from veeksha.config.microbenchmarks import MicrobenchmarkConfig
from veeksha.logger import init_logger
from veeksha.microbenchmarks.decode_profiler import DecodeProfiler
from veeksha.microbenchmarks.prefill_profiler import PrefillProfiler

logger = init_logger(__name__)


def run_microbenchmarks(config: MicrobenchmarkConfig):
    """Run enabled microbenchmark profilers based on config."""
    logger.info(f"Running microbenchmarks with output dir: {config.output_dir}")
    
    # Update benchmark config with the microbenchmark output directory
    from dataclasses import replace
    benchmark_config = replace(
        config.benchmark_config,
        metrics_config=replace(
            config.benchmark_config.metrics_config,
            output_dir=config.output_dir
        )
    )
    
    if config.prefill_profiler.enabled:
        logger.info("Running prefill profiler...")
        
        # Create embedded prefill config for compatibility with existing profiler
        prefill_config = config.create_prefill_profiler_config()
        
        # Create prefill-specific output directory
        prefill_output_dir = os.path.join(config.output_dir, "prefill")
        
        # Update benchmark config with prefill profiler config and output directory
        benchmark_config_with_prefill = replace(
            benchmark_config,
            prefill_profiler_config=prefill_config,
            metrics_config=replace(
                benchmark_config.metrics_config,
                output_dir=prefill_output_dir
            )
        )
        
        prefill_profiler = PrefillProfiler(benchmark_config_with_prefill)
        prefill_profiler.run()
        logger.info("Prefill profiler completed.")
    
    if config.decode_profiler.enabled:
        logger.info("Running decode profiler...")
        
        # Create embedded decode config for compatibility with existing profiler
        decode_config = config.create_decode_profiler_config()
        
        # Create decode-specific output directory
        decode_output_dir = os.path.join(config.output_dir, "decode")
        
        # Update benchmark config with decode profiler config and output directory
        benchmark_config_with_decode = replace(
            benchmark_config,
            decode_profiler_config=decode_config,
            metrics_config=replace(
                benchmark_config.metrics_config,
                output_dir=decode_output_dir
            )
        )
        
        decode_profiler = DecodeProfiler(benchmark_config_with_decode)
        decode_profiler.run()
        logger.info("Decode profiler completed.")
    
    if not config.prefill_profiler.enabled and not config.decode_profiler.enabled:
        logger.warning("No profilers enabled. Enable prefill_profiler or decode_profiler in config.")
    
    logger.info("Microbenchmark profiling completed.")


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    configs = MicrobenchmarkConfig.create_from_cli_args()
    for config in configs:
        config.write_config_to_file()
        run_microbenchmarks(config)