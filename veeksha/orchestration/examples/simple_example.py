"""
Simple example of running a benchmark with automatic server orchestration.

This example demonstrates the basic workflow:
1. Configure server (vLLM)
2. Use existing benchmark configuration (from CLI or code)
3. Run benchmark with automatic server lifecycle management

The server will be automatically launched, health-checked, used for
the benchmark, and then shut down.

Usage:
    # First create or adapt a benchmark config, then:
    python simple_example.py
"""

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.server import ServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration.benchmark_orchestrator import run_benchmark_with_server

logger = init_logger(__name__)


def main():
    """Run a simple benchmark with server orchestration."""
    
    logger.info("=" * 80)
    logger.info("Simple Microbenchmark Example")
    logger.info("=" * 80)
    
    # Step 1: Configure the server
    logger.info("Configuring vLLM server...")
    server_config = ServerConfig(
        engine="vllm",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        host="localhost",
        port=8000,
        api_key="test-key-123",
        tensor_parallel_size=1,
        dtype="auto",
        auto_shutdown=True,  # Automatically shutdown after benchmark
        startup_timeout=300,  # 5 minutes for model loading
    )
    
    # Step 2: Get benchmark configuration
    # Option A: Load from CLI args
    logger.info("Loading benchmark configuration...")
    benchmark_configs = BenchmarkConfig.create_from_cli_args()
    
    if not benchmark_configs:
        logger.error("No benchmark configuration provided!")
        logger.info("Please provide benchmark configuration via command-line arguments.")
        logger.info("Example:")
        logger.info("  python simple_example.py \\")
        logger.info("    --max_completed_requests 20 \\")
        logger.info("    --metrics_config_output_dir ./results")
        return
    
    benchmark_config = benchmark_configs[0]
    
    # Option B: Or create programmatically (see BenchmarkConfig documentation)
    # benchmark_config = BenchmarkConfig(...)
    
    # Step 3: Run the benchmark with automatic server management
    logger.info("Starting benchmark with server orchestration...")
    
    try:
        metrics = run_benchmark_with_server(
            benchmark_config=benchmark_config,
            server_config=server_config,
        )
        
        # Step 4: Display results
        logger.info("=" * 80)
        logger.info("Benchmark Results")
        logger.info("=" * 80)
        
        summary = metrics.get_aggregated_summary()
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
        
        logger.info(f"\nDetailed results saved to: {benchmark_config.metrics_config.output_dir}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
