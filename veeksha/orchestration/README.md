# Server Orchestration for Benchmarks

This module provides resource-aware server orchestration capabilities for running LLM inference workloads. It allows you to automatically launch, manage, and shut down inference servers (like vLLM) as part of your benchmarking workflow.

## Features

- **Automatic Server Lifecycle Management**: Launch, health check, and shutdown servers automatically
- **Multiple Workload Types**: Support for standard benchmarks, microbenchmarks, and lm_eval tasks
- **Resource Awareness**: Specify GPU allocation, tensor parallelism, and other resource constraints
- **Multiple Server Support**: Extensible architecture supports vLLM and can be extended to other systems
- **Parameter Sweeps**: Run benchmarks across multiple configurations efficiently
- **Context Manager Support**: Clean resource management with Python context managers

## Supported Workload Types

1. **Standard Benchmarks**: Traditional throughput/latency benchmarks with request generators
2. **Microbenchmarks**: Prefill and decode probes for performance profiling
3. **LM-Eval**: Run evaluation harness tasks (HellaSwag, MMLU, etc.)

## Quick Start

### Standard Benchmark Example

```python
from veeksha.config.server import ServerConfig
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.orchestration import managed_server
from veeksha.benchmark import run_benchmark

# Configure server
server_config = ServerConfig(
    engine="vllm",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    port=8000,
    tensor_parallel_size=1,
    auto_shutdown=True,
)

# Configure benchmark (use your existing benchmark config)
benchmark_config = BenchmarkConfig(...)

# Run benchmark with automatic server management
with managed_server(server_config) as info:
    print(f"Server ready at {info['api_base']}")
    metrics = run_benchmark(benchmark_config)
```

### LM-Eval Example

```python
import json
import os

from veeksha.benchmark import run_benchmark
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import ClientConfig
from veeksha.config.generators.request_generator.lmeval_generator import (
    LmevalRequestGeneratorConfig,
)
from veeksha.config.metrics import MetricsConfig
from veeksha.config.server import ServerConfig
from veeksha.orchestration import managed_server

# Configure server
server_config = ServerConfig(
    engine="vllm",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    port=8000,
    auto_shutdown=True,
)

# Configure lm_eval benchmark
benchmark_config = BenchmarkConfig(
    client_config=ClientConfig(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    request_generator_config=LmevalRequestGeneratorConfig(
        tasks=["hellaswag", "winogrande"],
        num_fewshot=5,
        limit=100,
    ),
    metrics_config=MetricsConfig(
        output_dir="./lmeval_results",
    ),
)

# Run lm_eval with automatic server management
with managed_server(server_config) as info:
    run_benchmark(benchmark_config)
    
    # Load results
    results_path = os.path.join(
        benchmark_config.metrics_config.output_dir, "lmeval_results.json"
    )
    with open(results_path) as f:
        results = json.load(f)
```

### API Usage

The recommended way to use server orchestration is through the Python API:

```python
from veeksha.config.server import ServerConfig
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.orchestration import managed_server
from veeksha.benchmark import run_benchmark

# Configure server
server_config = ServerConfig(
    engine="vllm",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    port=8000,
    tensor_parallel_size=1,
)

# Use your existing benchmark configuration
benchmark_config = BenchmarkConfig.create_from_cli_args()[0]

# Run benchmark with automatic server management
with managed_server(server_config) as info:
    metrics = run_benchmark(benchmark_config)
```

### Command-Line Integration

While a full command-line interface is being developed, you can integrate server orchestration
into your existing workflow by creating a Python script that:

1. Creates your benchmark config (using `BenchmarkConfig.create_from_cli_args()`)
2. Creates a server config
3. Uses `managed_server()` context manager to run the benchmark

See the examples in `veeksha/orchestration/examples/` for templates.

## Architecture

### Components

1. **ServerConfig** (`veeksha/config/server.py`): Configuration for server launch parameters
2. **BaseServerManager** (`veeksha/orchestration/server_manager.py`): Abstract base class for server management
3. **VLLMServerManager** (`veeksha/orchestration/vllm_server.py`): vLLM-specific implementation
4. **benchmark_orchestrator** (`veeksha/orchestration/benchmark_orchestrator.py`): High-level orchestration logic

### Server Lifecycle

```
┌─────────────────────────────────────────────────────┐
│  1. Configure Server                                 │
│     - Model, GPUs, ports, etc.                      │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  2. Launch Server                                    │
│     - Start process with configured parameters      │
│     - Set environment variables                     │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  3. Health Check & Wait for Ready                   │
│     - Poll /health endpoint                         │
│     - Wait until server responds                    │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  4. Run Benchmark                                    │
│     - Execute benchmark against server              │
│     - Collect metrics                               │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  5. Shutdown Server                                  │
│     - Graceful termination                          │
│     - Resource cleanup                              │
└─────────────────────────────────────────────────────┘
```

## ServerConfig Options

### Basic Options

- `engine`: Inference engine (`"vllm"`, etc.)
 - `engine`: Inference engine (`"vllm"`, "vajra", etc.)
- `model`: Model name or path
- `host`: Server host address (default: `"localhost"`)
- `port`: Server port (default: `8000`)
- `api_key`: API authentication key

### Resource Options

- `tensor_parallel_size`: Number of GPUs for tensor parallelism
- `gpu_ids`: Specific GPU IDs to use (e.g., `[0, 1, 3]`)
- `dtype`: Model data type (`"auto"`, `"float16"`, `"bfloat16"`)
- `max_model_len`: Maximum context length

### Lifecycle Options

- `startup_timeout`: Seconds to wait for server startup (default: 300)
- `health_check_interval`: Seconds between health checks (default: 2.0)
- `auto_shutdown`: Automatically shutdown after benchmark (default: `True`)

### Additional Arguments

Use `additional_args` dict for engine-specific options:

```python
server_config = ServerConfig(
    engine="vllm",  # or "vajra" to launch Vajra's OpenAI-compatible server
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    additional_args={
        "rope_scaling": {"type": "dynamic", "factor": 2.0},
        "max_num_seqs": 256,
        "gpu_memory_utilization": 0.9,
    }
)
```

## Use Cases

### 1. Resource-Constrained Environments

When you have limited GPUs and need to run many experiments:

```python
# Run experiments sequentially, reusing GPUs
configs = [
    ("model-A", 1), ("model-A", 2), ("model-A", 4),
    ("model-B", 1), ("model-B", 2), ("model-B", 4),
]

for model, tp_size in configs:
    server_config = ServerConfig(
        model=model,
        tensor_parallel_size=tp_size,
        gpu_ids=list(range(tp_size)),  # Use first N GPUs
        auto_shutdown=True,
    )
    
    # Server launches, runs benchmark, shuts down
    # GPUs freed for next configuration
    with managed_server(server_config) as info:
        run_benchmark(benchmark_config)
```

### 2. Microbenchmark Parameter Sweeps

Test performance across different tensor parallelism configurations:

```python
from veeksha.config.microbenchmark import MicrobenchmarkConfig, PrefillProbeConfig
from veeksha.orchestration import run_microbenchmark_with_server

model = "meta-llama/Meta-Llama-3-8B-Instruct"
probe_config = PrefillProbeConfig(
    prefill_lengths=[256, 512, 1024],
    num_requests_per_prefill_length=10,
)

for tp_size in [1, 2, 4]:
    server_config = ServerConfig(
        model=model,
        tensor_parallel_size=tp_size,
        port=8000 + tp_size,
        gpu_ids=list(range(tp_size)),
        auto_shutdown=True,
    )
    
    microbenchmark_config = MicrobenchmarkConfig(
        model=model,
        output_dir=f"./results/tp{tp_size}",
        probe_config=probe_config,
    )
    
    run_microbenchmark_with_server(
        microbenchmark_config=microbenchmark_config,
        server_config=server_config,
    )
```

### 3. Model Evaluation Campaigns

Evaluate multiple models on standard benchmarks:

```python
import json
import os

from veeksha.benchmark import run_benchmark
from veeksha.orchestration import managed_server

models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
]

for i, model in enumerate(models):
    server_config = ServerConfig(
        model=model,
        port=8000 + i,
        auto_shutdown=True,
    )
    
    benchmark_config = BenchmarkConfig(
        client_config=ClientConfig(model=model),
        request_generator_config=LmevalRequestGeneratorConfig(
            tasks=["hellaswag", "winogrande", "arc_easy"],
            num_fewshot=5,
        ),
        metrics_config=MetricsConfig(
            output_dir=f"./eval_results/{model.replace('/', '_')}",
        ),
    )
    
    with managed_server(server_config) as info:
        run_benchmark(benchmark_config)
        
        # Load results
        results_path = os.path.join(
            benchmark_config.metrics_config.output_dir, "lmeval_results.json"
        )
        with open(results_path) as f:
            results = json.load(f)
        print(f"{model}: {results}")
```

### 3. Manual Server Management

If you need more control:

```python
from veeksha.orchestration.vllm_server import VLLMServerManager

# Create and manage server manually
server_manager = VLLMServerManager(server_config)

try:
    server_manager.launch()
    server_manager.wait_for_ready()
    
    # Run multiple benchmarks against same server
    for config in benchmark_configs:
        run_benchmark(config)
        
finally:
    server_manager.shutdown()
```

### 4. Context Manager Pattern

```python
from veeksha.orchestration.vllm_server import VLLMServerManager

with VLLMServerManager(server_config) as server:
    # Server is launched and ready
    run_benchmark(benchmark_config)
    # Server automatically shut down on exit
```

## Extending to Other Inference Systems

To add support for a new inference system:

1. Create a new manager class inheriting from `BaseServerManager`
2. Implement `_build_launch_command()` method
3. Optionally override `health_check()` if needed

Example for a hypothetical system:

```python
from veeksha.orchestration.server_manager import BaseServerManager

class TGIServerManager(BaseServerManager):
    def _build_launch_command(self) -> List[str]:
        return [
            "text-generation-launcher",
            "--model-id", self.config.model,
            "--port", str(self.config.port),
            "--num-shard", str(self.config.tensor_parallel_size),
            # ... other TGI-specific args
        ]
```

## Examples

See `veeksha/orchestration/examples/` for complete working examples:

- `simple_example.py`: Basic single benchmark with server orchestration
- `microbenchmark_orchestration.py`: Run prefill/decode probes with orchestration
- `lmeval_orchestration.py`: Run lm_eval tasks with orchestration

Each example demonstrates:
- How to configure servers and workloads
- Parameter sweep patterns
- Error handling and best practices

## Troubleshooting

### Server fails to start

- Check GPU availability: `nvidia-smi`
- Verify model access (HuggingFace login if needed)
- Increase `startup_timeout` for large models
- Check server logs via `server_manager.get_server_logs()`

### Port conflicts

- Use different ports for concurrent servers
- Check for existing processes: `lsof -i :8000`

### Resource issues

- Verify GPU memory with `nvidia-smi`
- Adjust `tensor_parallel_size` based on available GPUs
- Use `gpu_ids` to explicitly assign GPUs

## Future Enhancements

Planned improvements (not yet implemented):

- [ ] Support for TGI, SGLang, and other inference systems
- [ ] Advanced resource scheduling for parallel experiments
- [ ] Persistent server pools
- [ ] Remote server management
- [ ] Integration with job schedulers (Slurm, etc.)

## Related

- Main benchmark documentation: [README.md](../../README.md)
- Capacity search: [veeksha/capacity_search/](../capacity_search/)
