# Server Orchestration for Microbenchmarks

This module provides resource-aware server orchestration capabilities for running LLM inference benchmarks. It allows you to automatically launch, manage, and shut down inference servers (like vLLM) as part of your benchmarking workflow.

## Features

- **Automatic Server Lifecycle Management**: Launch, health check, and shutdown servers automatically
- **Resource Awareness**: Specify GPU allocation, tensor parallelism, and other resource constraints
- **Multiple Server Support**: Extensible architecture supports vLLM and can be extended to other systems
- **Parameter Sweeps**: Run benchmarks across multiple configurations efficiently
- **Context Manager Support**: Clean resource management with Python context managers

## Quick Start

### Basic Example

```python
from veeksha.config.server import ServerConfig
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.orchestration.benchmark_orchestrator import run_benchmark_with_server

# Configure server
server_config = ServerConfig(
    engine="vllm",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    port=8000,
    tensor_parallel_size=1,
)

# Configure benchmark (use your existing benchmark config)
benchmark_config = BenchmarkConfig(...)

# Run benchmark with automatic server management
metrics = run_benchmark_with_server(
    benchmark_config=benchmark_config,
    server_config=server_config,
)
```

### API Usage

The recommended way to use server orchestration is through the Python API:

```python
from veeksha.config.server import ServerConfig
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.orchestration.benchmark_orchestrator import run_benchmark_with_server

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
metrics = run_benchmark_with_server(
    benchmark_config=benchmark_config,
    server_config=server_config,
)
```

### Command-Line Integration

While a full command-line interface is being developed, you can integrate server orchestration
into your existing workflow by creating a Python script that:

1. Creates your benchmark config (using `BenchmarkConfig.create_from_cli_args()`)
2. Creates a server config
3. Calls `run_benchmark_with_server()`

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
    engine="vllm",
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
    )
    
    # Server launches, runs benchmark, shuts down
    # GPUs freed for next configuration
    run_benchmark_with_server(benchmark_config, server_config)
```

### 2. Microbenchmark Sweeps

Test different server parameters:

```python
# Sweep batch sizes
for max_seqs in [128, 256, 512]:
    server_config = ServerConfig(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        additional_args={"max_num_seqs": max_seqs}
    )
    run_benchmark_with_server(benchmark_config, server_config)
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
- More examples coming soon!

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
