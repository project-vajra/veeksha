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
from veeksha.orchestration.benchmark_orchestrator import create_server_manager
from veeksha.microbenchmark import Microbenchmark

logger = init_logger(__name__)


def example_prefill_probe(api_url: str, api_key: str):
    """Run prefill probe with common server."""
    
    logger.info("=" * 80)
    logger.info("Example: Prefill Probe with Common Server")
    logger.info("=" * 80)
    
    # Configure microbenchmark with prefill probe
    microbenchmark_config = MicrobenchmarkConfig(
        model="Qwen/Qwen3-1.7B",
        api_url=api_url,
        api_key=api_key,
        output_dir="./microbenchmark_results/prefill_probe",
        probe_config=PrefillProbeConfig(
            prefill_lengths=[128, 256, 512, 1024],
            num_requests_per_prefill_length=10,
        ),
        timeout=600,
    )
    
    # Run directly
    microbenchmark = Microbenchmark(microbenchmark_config)
    microbenchmark.run()
    
    logger.info("Prefill probe completed!")


def example_decode_probe(api_url: str, api_key: str):
    """Run decode probe with common server."""
    
    logger.info("=" * 80)
    logger.info("Example: Decode Probe with Common Server")
    logger.info("=" * 80)
    
    # Configure microbenchmark with decode probe
    microbenchmark_config = MicrobenchmarkConfig(
        model="Qwen/Qwen3-1.7B",
        api_url=api_url,
        api_key=api_key,
        output_dir="./microbenchmark_results/decode_probe",
        probe_config=DecodeProbeConfig(
            context_lengths=[128, 512],
            batch_sizes=[1, 4, 8],
            profiling_iterations=20,
        ),
        timeout=600,
    )
    
    # Run directly
    microbenchmark = Microbenchmark(microbenchmark_config)
    microbenchmark.run()
    
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
    
    if len(sys.argv) > 2:
        example = sys.argv[2]
        if example == "prefill":
            # Start server for prefill (or use existing)
            server_config = ServerConfig(
                engine="vllm",
                model="Qwen/Qwen3-1.7B",
                host="localhost",
                port=8000,
                tensor_parallel_size=1,
                auto_shutdown=False,
            )
            server_manager = create_server_manager(server_config)
            try:
                server_manager.launch()
                server_manager.wait_for_ready()
                logger.info("Server started for prefill")
            except Exception as e:
                logger.info("Server already running on port 8000, using existing server")
            example_prefill_probe(server_config.get_api_base_url(), server_config.api_key)
        elif example == "decode":
            # Start server for decode (or use existing)
            server_config = ServerConfig(
                engine="vllm",
                model="Qwen/Qwen3-1.7B",
                host="localhost",
                port=8000,
                tensor_parallel_size=1,
                auto_shutdown=False,
            )
            server_manager = create_server_manager(server_config)
            try:
                server_manager.launch()
                server_manager.wait_for_ready()
                logger.info("Server started for decode")
            except Exception as e:
                logger.info("Server already running on port 8000, using existing server")
            example_decode_probe(server_config.get_api_base_url(), server_config.api_key)
        elif example == "sweep":
            example_parameter_sweep()
        else:
            logger.error(f"Unknown example: {example}")
            logger.info("Available examples: prefill, decode, sweep")
    else:
        logger.info("Running all examples...")
        logger.info("To run a specific example: python microbenchmark_orchestration.py [prefill|decode|sweep]")
        logger.info("")
        
        # Start common server for prefill and decode
        common_server_config = ServerConfig(
            engine="vllm",
            model="Qwen/Qwen3-1.7B",
            host="localhost",
            port=8000,
            tensor_parallel_size=1,
            auto_shutdown=False,
        )
        server_manager = create_server_manager(common_server_config)
        server_manager.launch()
        server_manager.wait_for_ready()
        logger.info("Common server ready at {}".format(common_server_config.get_api_base_url()))
        
        example_prefill_probe(common_server_config.get_api_base_url(), common_server_config.api_key)
        logger.info("\n" * 2)
        
        example_decode_probe(common_server_config.get_api_base_url(), common_server_config.api_key)
        logger.info("\n" * 2)
        
        # Shutdown common server
        server_manager.shutdown()
        logger.info("Common server shut down.")
        logger.info("\n" * 2)
        
        example_parameter_sweep()


if __name__ == "__main__":
    main()
