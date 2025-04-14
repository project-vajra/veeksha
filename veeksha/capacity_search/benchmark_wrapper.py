"""
This file contains the wrapper for the benchmarking.
"""

import os
import subprocess
import signal
import time

from veeksha.capacity_search.config.config import BenchmarkConfig, JobConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


def setup_api_environment(
    openai_api_key=None,
    openai_api_url=None,
):
    """Set up environment variables for OpenAI API"""
    assert openai_api_key is not None, "OpenAI API key is required"
    assert openai_api_url is not None, "OpenAI port is required"
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_API_BASE"] = openai_api_url


def run(
    job_config: JobConfig,
    benchmark_config: BenchmarkConfig,
):
    """Main function to run benchmark"""

    setup_api_environment(
        openai_api_key=job_config.server_config.openai_api_key,
        openai_api_url=job_config.server_config.openai_api_url,
    )

    benchmark_command = f"/scratch/chus/repos/envs/env-sglang/bin/python -m veeksha.run_benchmark {job_config.to_args()} {benchmark_config.to_args()}"
    logger.info(f"Running benchmark with command: {benchmark_command}")
    
    # Start the benchmark process
    benchmark_process = subprocess.Popen(benchmark_command, shell=True, preexec_fn=os.setsid)
    
    # Use a timeout that's slightly longer than the benchmark_config timeout
    # to allow for graceful termination
    timeout = getattr(benchmark_config, "timeout", -1)
    if timeout > 0:
        timeout += 10  # Add 10 seconds buffer
    
    start_time = time.time()
    try:
        # Wait with timeout checking
        while benchmark_process.poll() is None:
            time.sleep(0.1)
            # Check if we've exceeded the timeout
            if timeout > 0 and time.time() - start_time > timeout:
                logger.warning(f"Benchmark exceeded timeout of {timeout}s, terminating...")
                # Kill the entire process group to ensure child processes are terminated
                os.killpg(os.getpgid(benchmark_process.pid), signal.SIGTERM)
                # Give it a moment to terminate gracefully
                time.sleep(2)
                # Force kill if still running
                if benchmark_process.poll() is None:
                    logger.warning("Benchmark did not terminate gracefully, force killing...")
                    os.killpg(os.getpgid(benchmark_process.pid), signal.SIGKILL)
                break
        
        # Check the final status
        exit_code = benchmark_process.poll()
        if exit_code is not None and exit_code != 0:
            logger.warning(f"Benchmark exited with non-zero status: {exit_code}")
        else:
            logger.info("Benchmark finished successfully")
            
    except KeyboardInterrupt:
        logger.warning("Received keyboard interrupt, terminating benchmark...")
        os.killpg(os.getpgid(benchmark_process.pid), signal.SIGTERM)
        raise
    except Exception as e:
        logger.error(f"Error while running benchmark: {e}")
        if benchmark_process.poll() is None:
            os.killpg(os.getpgid(benchmark_process.pid), signal.SIGTERM)
        raise
