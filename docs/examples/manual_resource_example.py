"""
Example: Automatic resource management for fine-grained control.

This example shows how server managers automatically allocate GPUs
when gpu_ids is not specified, using the ResourceManager for intelligent
resource scheduling.
"""

import time

from veeksha.benchmark import run_benchmark
from veeksha.config.benchmark import (
    BenchmarkConfig,
    ClientConfig,
    MetricsConfig,
    SyntheticRequestGeneratorConfig,
)
from veeksha.config.generators.interval_generator.poisson_generator import (
    PoissonRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.length_generator.trace_generator import (
    TraceRequestLengthGeneratorConfig,
)
from veeksha.config.server import ServerConfig
from veeksha.orchestration import ResourceManager
from veeksha.orchestration.benchmark_orchestrator import managed_server


def run_experiment_with_resources(
    model: str,
    tp_size: int,
    port: int,
    resource_manager: ResourceManager,
    job_id: str,
):
    """Run a single experiment with automatic resource management."""
    print(f"\n{'='*60}")
    print(f"Starting: {model} (TP={tp_size})")
    print(f"Job ID: {job_id}")

    try:
        # Create server config - let server manager auto-allocate GPUs
        server_config = ServerConfig(
            engine="vajra",
            model=model,
            port=port,
            tensor_parallel_size=tp_size,
            gpu_ids=None,  # Let server manager auto-allocate
            auto_shutdown=True,
        )
        # Create benchmark config
        benchmark_config = BenchmarkConfig(
            api_url=f"http://localhost:{port}/v1",
            timeout=300,
            max_completed_requests=20,
            client_config=ClientConfig(model=model),
            request_generator_config=SyntheticRequestGeneratorConfig(
                interval_generator_config=PoissonRequestIntervalGeneratorConfig(
                    qps=0.5
                ),
                length_generator_config=TraceRequestLengthGeneratorConfig(
                    trace_file="./veeksha/data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv",
                    max_tokens=8192,
                ),
            ),
            metrics_config=MetricsConfig(output_dir=f"results/{job_id}"),
        )

        # Use managed_server for automatic lifecycle management
        print("Launching server...")
        with managed_server(server_config) as server_info:
            print(f"Server ready at {server_info['api_base']}! Running benchmark...")

            # Run benchmark
            result = run_benchmark(benchmark_config)

            print(f"Benchmark complete: {result.output_dir}")

        return result

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    """Demonstrate automatic resource management."""
    print("Automatic Resource Management Example")
    print("=" * 60)

    # Initialize resource manager
    resource_manager = ResourceManager(detect_gpus=True)

    # Show initial resource status
    status = resource_manager.get_resource_status()
    print(f"\nInitial GPU Status:")
    print(f"  Total GPUs: {status['total_gpus']}")
    print(f"  Free GPUs:  {status['free_gpus']}")
    print()

    # Define experiments
    experiments = [
        ("meta-llama/Meta-Llama-3-8B-Instruct", 1, 8000),
        ("meta-llama/Meta-Llama-3-8B-Instruct", 2, 8000),
        ("Qwen/Qwen2.5-0.5B-Instruct", 1, 8000),
        ("Qwen/Qwen2.5-0.5B-Instruct", 2, 8000),
    ]

    results = []

    # Run experiments sequentially
    for idx, (model, tp_size, port) in enumerate(experiments):
        job_id = f"exp_{idx}_{model.split('/')[-1]}_tp{tp_size}"

        result = run_experiment_with_resources(
            model=model,
            tp_size=tp_size,
            port=port,
            resource_manager=resource_manager,
            job_id=job_id,
        )

        results.append(result)

        # Show current resource status
        status = resource_manager.get_resource_status()
        print(f"\nCurrent GPU Status:")
        print(f"  Free GPUs: {status['free_gpus']}/{status['total_gpus']}")
        print(f"  Active jobs: {status['active_jobs']}")

        # Brief pause between experiments
        time.sleep(2)

    # Final summary
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print(f"Successful: {sum(1 for r in results if r is not None)}/{len(results)}")

    # Final resource status
    status = resource_manager.get_resource_status()
    print(f"\nFinal GPU Status:")
    print(f"  Free GPUs: {status['free_gpus']}/{status['total_gpus']}")
    print(f"  Active jobs: {status['active_jobs']}")


if __name__ == "__main__":
    main()
