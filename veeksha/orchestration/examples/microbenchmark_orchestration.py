"""
Example: Running microbenchmarks with automatic server orchestration.

This example demonstrates how to run prefill and decode probes with
automatic server lifecycle management:
1. Launch vLLM server
2. Run microbenchmark probes (prefill/decode)
3. Automatically shutdown server

This is useful for:
- Profiling different models without manual server management
- Running parameter sweeps across different TP sizes
- Collecting performance baselines
"""

from veeksha.config.microbenchmark import MicrobenchmarkConfig, PrefillProbeConfig, DecodeProbeConfig
from veeksha.config.server import ServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration import run_microbenchmark_with_server

logger = init_logger(__name__)


def example_prefill_probe():
    """Run prefill probe with automatic server orchestration."""
    
    logger.info("=" * 80)
    logger.info("Example: Prefill Probe with Server Orchestration")
    logger.info("=" * 80)
    
    # Configure server
    server_config = ServerConfig(
        engine="vllm",
        model="Qwen/Qwen3-1.7B",
        host="localhost",
        port=8000,
        tensor_parallel_size=1,
        auto_shutdown=True,
        startup_timeout=300,
    )
    
    # Configure microbenchmark with prefill probe
    microbenchmark_config = MicrobenchmarkConfig(
        model="Qwen/Qwen3-1.7B",
        output_dir="./microbenchmark_results/prefill_probe",
        probe_config=PrefillProbeConfig(
            prefill_lengths=[128, 256, 512, 1024],
            num_requests_per_prefill_length=10,
        ),
        timeout=600,
    )
    
    # Run with orchestration
    run_microbenchmark_with_server(
        microbenchmark_config=microbenchmark_config,
        server_config=server_config,
    )
    
    logger.info("Prefill probe completed!")


def example_decode_probe():
    """Run decode probe with automatic server orchestration."""
    
    logger.info("=" * 80)
    logger.info("Example: Decode Probe with Server Orchestration")
    logger.info("=" * 80)
    
    # Configure server
    server_config = ServerConfig(
        engine="vllm",
        model="Qwen/Qwen3-1.7B",
        host="localhost",
        port=8001,
        tensor_parallel_size=1,
        auto_shutdown=True,
        startup_timeout=300,
    )
    
    # Configure microbenchmark with decode probe
    microbenchmark_config = MicrobenchmarkConfig(
        model="Qwen/Qwen3-1.7B",
        output_dir="./microbenchmark_results/decode_probe",
        probe_config=DecodeProbeConfig(
            context_lengths=[128, 512],
            batch_sizes=[1, 4, 8],
            profiling_iterations=20,
        ),
        timeout=600,
    )
    
    # Run with orchestration
    run_microbenchmark_with_server(
        microbenchmark_config=microbenchmark_config,
        server_config=server_config,
    )
    
    logger.info("Decode probe completed!")


def example_parameter_sweep():
    """Run microbenchmarks across different tensor parallelism sizes."""
    
    logger.info("=" * 80)
    logger.info("Example: Parameter Sweep Across TP Sizes")
    logger.info("=" * 80)
    
    model = "Qwen/Qwen3-1.7B"
    tp_sizes = [1, 2, 4]
    
    probe_config = PrefillProbeConfig(
        prefill_lengths=[256, 512, 1024],
        num_requests_per_prefill_length=10,
    )
    
    for tp_size in tp_sizes:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Running with TP={tp_size}")
        logger.info(f"{'=' * 80}")
        
        # Create server config for this TP size
        server_config = ServerConfig(
            engine="vllm",
            model=model,
            host="localhost",
            port=8000 + tp_size,  # Different port for each
            tensor_parallel_size=tp_size,
            gpu_ids=list(range(tp_size)),
            auto_shutdown=True,
            startup_timeout=300,
        )
        
        # Create microbenchmark config
        microbenchmark_config = MicrobenchmarkConfig(
            model=model,
            output_dir=f"./microbenchmark_results/tp_sweep/tp{tp_size}",
            probe_config=probe_config,
            timeout=600,
        )
        
        try:
            run_microbenchmark_with_server(
                microbenchmark_config=microbenchmark_config,
                server_config=server_config,
            )
            logger.info(f"✓ TP={tp_size} completed successfully")
        except Exception as e:
            logger.error(f"✗ TP={tp_size} failed: {e}")
            continue
    
    logger.info("\nParameter sweep completed!")


def main():
    """Run example based on command line argument or run all."""
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "prefill":
            example_prefill_probe()
        elif example == "decode":
            example_decode_probe()
        elif example == "sweep":
            example_parameter_sweep()
        else:
            logger.error(f"Unknown example: {example}")
            logger.info("Available examples: prefill, decode, sweep")
    else:
        logger.info("Running all examples...")
        logger.info("To run a specific example: python microbenchmark_orchestration.py [prefill|decode|sweep]")
        logger.info("")
        
        example_prefill_probe()
        logger.info("\n" * 2)
        
        example_decode_probe()
        logger.info("\n" * 2)
        
        example_parameter_sweep()


if __name__ == "__main__":
    main()
