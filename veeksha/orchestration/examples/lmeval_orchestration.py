"""
Example: Running lm_eval tasks with automatic server orchestration.

This example demonstrates how to run lm_eval evaluations with
automatic server lifecycle management:
1. Launch vLLM server
2. Run lm_eval tasks
3. Automatically shutdown server

This is useful for:
- Evaluating different models on standard benchmarks
- Running evaluations across different server configurations
- Automating evaluation workflows
"""

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import ClientConfig
from veeksha.config.generators.request_generator.lmeval_generator import (
    LmevalRequestGeneratorConfig,
)
from veeksha.config.metrics import MetricsConfig
from veeksha.config.server import ServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration import run_lmeval_with_server

logger = init_logger(__name__)


def example_hellaswag():
    """Run HellaSwag benchmark with automatic server orchestration."""
    
    logger.info("=" * 80)
    logger.info("Example: HellaSwag Benchmark with Server Orchestration")
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
    
    # Configure lm_eval benchmark
    benchmark_config = BenchmarkConfig(
        seed=42,
        timeout=1800,  # 30 minutes
        max_completed_requests=100,  # Limit for demo
        client_config=ClientConfig(
            model="Qwen/Qwen3-1.7B",
            num_clients=1,
        ),
        request_generator_config=LmevalRequestGeneratorConfig(
            tasks=["hellaswag"],
            num_fewshot=5,
            limit=100,  # Evaluate on subset for demo
            is_logit_based=True,  # HellaSwag is a multiple choice task
        ),
        metrics_config=MetricsConfig(
            output_dir="./lmeval_results/hellaswag",
        ),
    )
    
    # Run with orchestration
    results = run_lmeval_with_server(
        benchmark_config=benchmark_config,
        server_config=server_config,
    )
    
    logger.info("HellaSwag evaluation completed!")
    logger.info(f"Results: {results}")


def example_multiple_tasks():
    """Run multiple lm_eval tasks with automatic server orchestration."""
    
    logger.info("=" * 80)
    logger.info("Example: Multiple Tasks with Server Orchestration")
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
    
    # Configure lm_eval benchmark with multiple tasks
    benchmark_config = BenchmarkConfig(
        seed=42,
        timeout=3600,  # 1 hour
        max_completed_requests=500,
        client_config=ClientConfig(
            model="Qwen/Qwen3-1.7B",
            num_clients=1,
        ),
        request_generator_config=LmevalRequestGeneratorConfig(
            tasks=["hellaswag", "winogrande", "arc_easy"],
            num_fewshot=5,
            limit=50,  # Evaluate on subset for demo
        ),
        metrics_config=MetricsConfig(
            output_dir="./lmeval_results/multiple_tasks",
        ),
    )
    
    # Run with orchestration
    results = run_lmeval_with_server(
        benchmark_config=benchmark_config,
        server_config=server_config,
    )
    
    logger.info("Multi-task evaluation completed!")
    
    # Display summary of results
    if "results" in results:
        logger.info("\nTask Results:")
        for task, metrics in results["results"].items():
            logger.info(f"  {task}:")
            for metric, value in metrics.items():
                logger.info(f"    {metric}: {value}")


def example_model_comparison():
    """Compare multiple models on the same task."""
    
    logger.info("=" * 80)
    logger.info("Example: Model Comparison")
    logger.info("=" * 80)
    
    models = [
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen2-7B-Instruct",
    ]
    
    all_results = {}
    
    for i, model in enumerate(models):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluating model: {model}")
        logger.info(f"{'=' * 80}")
        
        # Create server config for this model
        server_config = ServerConfig(
            engine="vllm",
            model=model,
            host="localhost",
            port=8000 + i,  # Different port for each
            tensor_parallel_size=1,
            auto_shutdown=True,
            startup_timeout=300,
        )
        
        # Create benchmark config
        benchmark_config = BenchmarkConfig(
            seed=42,
            timeout=1800,
            max_completed_requests=100,
            client_config=ClientConfig(
                model=model,
                num_clients=1,
            ),
            request_generator_config=LmevalRequestGeneratorConfig(
                tasks=["hellaswag"],
                num_fewshot=5,
                limit=50,
            ),
            metrics_config=MetricsConfig(
                output_dir=f"./lmeval_results/comparison/{model.replace('/', '_')}",
            ),
        )
        
        try:
            results = run_lmeval_with_server(
                benchmark_config=benchmark_config,
                server_config=server_config,
            )
            all_results[model] = results
            logger.info(f"✓ {model} completed successfully")
        except Exception as e:
            logger.error(f"✗ {model} failed: {e}")
            continue
    
    # Print comparison
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 80)
    for model, results in all_results.items():
        logger.info(f"\n{model}:")
        if "results" in results:
            for task, metrics in results["results"].items():
                logger.info(f"  {task}:")
                for metric, value in metrics.items():
                    logger.info(f"    {metric}: {value}")


def main():
    """Run example based on command line argument or run all."""
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "hellaswag":
            example_hellaswag()
        elif example == "multiple":
            example_multiple_tasks()
        elif example == "comparison":
            example_model_comparison()
        else:
            logger.error(f"Unknown example: {example}")
            logger.info("Available examples: hellaswag, multiple, comparison")
    else:
        logger.info("Running all examples...")
        logger.info("To run a specific example: python lmeval_orchestration.py [hellaswag|multiple|comparison]")
        logger.info("")
        
        example_hellaswag()
        logger.info("\n" * 2)
        
        example_multiple_tasks()
        logger.info("\n" * 2)
        
        example_model_comparison()


if __name__ == "__main__":
    main()
