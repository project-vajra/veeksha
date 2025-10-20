Server Orchestration for Benchmarks
===================================

This module provides comprehensive server orchestration and resource management capabilities for running LLM inference workloads. It allows you to automatically launch, manage, and shut down inference servers (like vLLM) with intelligent GPU allocation across multiple experiments.

Features
--------

Core Orchestration
^^^^^^^^^^^^^^^^^^

- **Automatic Server Lifecycle Management**: Launch, health check, and shutdown servers automatically
- **Multiple Workload Types**: Support for standard benchmarks, microbenchmarks, and lm_eval tasks
- **Multiple Server Support**: Extensible architecture supports vLLM, Vajra, and can be extended to other systems
- **Context Manager Support**: Clean resource management with Python context managers

Resource Management
^^^^^^^^^^^^^^^^^^^

- **Automatic GPU Detection**: Detect available GPUs using nvidia-smi
- **Intelligent GPU Allocation**: Allocate GPUs efficiently across experiments
- **Contiguous GPU Allocation**: Ensure contiguous GPU IDs for better performance
- **Resource Tracking**: Monitor GPU utilization and active jobs
- **Wait for Resources**: Queue jobs until resources become available
- **Multi-Node Support**: Allocate resources across multiple compute nodes

Architecture
------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                         VEEKSHA RESOURCE MANAGEMENT                      │
   └─────────────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                          MID-LEVEL RUNNERS                               │
   ├─────────────────────────────────────────────────────────────────────────┤
   │                                                                          │
   │  ParallelBenchmarkRunner            SequentialJobQueue                 │
   │  ┌─────────────────────┐            ┌──────────────────┐              │
   │  │ • Thread Pool       │            │ • FIFO Queue     │              │
   │  │ • Concurrent Exec   │            │ • Sequential Exec│              │
   │  │ • Max Workers       │            │ • Reproducible   │              │
   │  └─────────────────────┘            └──────────────────┘              │
   │           │                                  │                          │
   │           └──────────────┬───────────────────┘                          │
   │                          │                                              │
   └──────────────────────────┼──────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                       RESOURCE MANAGER (CORE)                            │
   ├─────────────────────────────────────────────────────────────────────────┤
   │                                                                          │
   │  ResourceManager                                                        │
   │  ┌─────────────────────────────────────────────────────────────┐       │
   │  │                                                              │       │
   │  │  GPU Detection              GPU Allocation                  │       │
   │  │  ┌──────────────┐           ┌──────────────┐               │       │
   │  │  │ nvidia-smi   │──────────▶│ Contiguous   │               │       │
   │  │  │ Auto-detect  │           │ Multi-node   │               │       │
   │  │  └──────────────┘           └──────────────┘               │       │
   │  │                                     │                       │       │
   │  │                                     ▼                       │       │
   │  │  Resource Tracking         Wait for Resources              │       │
   │  │  ┌──────────────┐          ┌──────────────┐               │       │
   │  │  │ Free GPUs    │          │ Timeout      │               │       │
   │  │  │ Allocated    │          │ Polling      │               │       │
   │  │  │ Active Jobs  │          │ Queue Jobs   │               │       │
   │  │  └──────────────┘          └──────────────┘               │       │
   │  │                                                             │       │
   │  └─────────────────────────────────────────────────────────────┘       │
   │                                                                          │
   └──────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                      SERVER MANAGEMENT LAYER                             │
   ├─────────────────────────────────────────────────────────────────────────┤
   │                                                                          │
   │  Enhanced ServerConfig              BaseServerManager                   │
   │  ┌─────────────────────┐            ┌──────────────────┐              │
   │  │ • gpu_ids           │            │ • launch()       │              │
   │  │ • priority          │            │ • wait_ready()   │              │
   │  │ • contiguous_gpus   │───────────▶│ • shutdown()     │              │
   │  │ • memory_estimate   │            │ • health_check() │              │
   │  └─────────────────────┘            └──────────────────┘              │
   │                                              │                          │
   │                                              ▼                          │
   │                          ┌────────────────────────────┐                │
   │                          │  VLLMServerManager         │                │
   │                          │  VajraServerManager        │                │
   │                          └────────────────────────────┘                │
   │                                                                          │
   └──────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                         BENCHMARK EXECUTION                              │
   ├─────────────────────────────────────────────────────────────────────────┤
   │                                                                          │
   │  run_benchmark()           run_capacity_search()      lm_eval           │
   │  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐ │
   │  │ Standard     │          │ SLO-based    │          │ Task Eval    │ │
   │  │ Benchmarks   │          │ Search       │          │ HellaSwag    │ │
   │  └──────────────┘          └──────────────┘          └──────────────┘ │
   │                                                                          │
   └─────────────────────────────────────────────────────────────────────────┘

Components
^^^^^^^^^^

1. **ServerConfig** (:code:`veeksha/config/server.py`): Configuration for server launch parameters
2. **BaseServerManager** (:code:`veeksha/orchestration/server_manager.py`): Abstract base class for server management
3. **VLLMServerManager** (:code:`veeksha/orchestration/vllm_server.py`): vLLM-specific implementation
4. **benchmark_orchestrator** (:code:`veeksha/orchestration/benchmark_orchestrator.py`): High-level orchestration logic

Server Lifecycle
^^^^^^^^^^^^^^^^

.. code-block:: text

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

Workflow Example
^^^^^^^^^^^^^^^^

1. User creates configs manually or programmatically

2. For each config combination:
   - Allocate resources with ResourceManager
   - Launch server with managed_server()
   - Run benchmark
   - Auto cleanup

3. ParallelBenchmarkRunner:
   - Sorts by GPU requirement
   - Submits to thread pool

4. For each job:

   .. code-block:: text

      ┌─────────────────────────────────────────────┐
      │ ResourceManager.wait_for_resources(2)       │
      │   ↓                                         │
      │ Allocates GPUs: [0, 1]                      │
      │   ↓                                         │
      │ ServerConfig.gpu_ids = [0, 1]              │
      │   ↓                                         │
      │ ServerManager.launch()                      │
      │   ↓                                         │
      │ run_benchmark(config)                       │
      │   ↓                                         │
      │ ServerManager.shutdown()                    │
      │   ↓                                         │
      │ ResourceManager.release_resources()         │
      └─────────────────────────────────────────────┘

5. Next job uses freed GPUs

6. All results collected and returned

Design Patterns
^^^^^^^^^^^^^^^

1. **LAYERED ARCHITECTURE**: High-level → Mid-level → Low-level. Simple APIs built on flexible primitives

2. **RESOURCE ACQUISITION IS INITIALIZATION (RAII)**: Automatic cleanup on context exit

3. **CONTEXT MANAGERS**: :code:`with managed_server(config):` provides automatic lifecycle management

4. **FACTORY PATTERN**: :code:`create_server_manager(config)` → specific implementation

5. **STRATEGY PATTERN**: Different allocation strategies (contiguous, fragmented, etc.)

Quick Start
-----------

Installation
^^^^^^^^^^^^

No additional dependencies needed - resource management is built into Veeksha.

Check GPU Availability
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from veeksha.orchestration import ResourceManager

   rm = ResourceManager(detect_gpus=True)
   status = rm.get_resource_status()

   print(f"Total GPUs: {status['total_gpus']}")
   print(f"Free GPUs: {status['free_gpus']}")
   print(f"Active jobs: {status['active_jobs']}")

Comprehensive Examples
----------------------

Example 1: Manual Resource Control with Parallel Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For custom scheduling logic with multiple experiments:

.. code-block:: python

   from veeksha.orchestration import ResourceManager, ParallelBenchmarkRunner
   from veeksha.orchestration.benchmark_orchestrator import managed_server
   from veeksha.config.server import ServerConfig
   from veeksha.config.benchmark import BenchmarkConfig
   from veeksha.benchmark import run_benchmark

   # Initialize resource manager
   rm = ResourceManager(detect_gpus=True)

   # Create experiment configurations
   experiments = [
       {
           "model": "meta-llama/Meta-Llama-3-8B-Instruct",
           "tensor_parallel_size": 1,
           "port": 8000,
           "job_id": "exp1"
       },
       {
           "model": "Qwen/Qwen2.5-0.5B-Instruct", 
           "tensor_parallel_size": 2,
           "port": 8001,
           "job_id": "exp2"
       }
   ]

   # Prepare configs for parallel execution
   server_configs = []
   benchmark_configs = []

   for exp in experiments:
       # Create server config
       server_config = ServerConfig(
           model=exp["model"],
           tensor_parallel_size=exp["tensor_parallel_size"],
           port=exp["port"],
           auto_shutdown=True,
       )
       
       # Create benchmark config (customize as needed)
       benchmark_config = BenchmarkConfig(
           max_completed_requests=50,
           timeout=600,
           client_config={"model": exp["model"]},
           request_generator_config={
               "type": "synthetic",
               "length_generator_config": {
                   "type": "fixed",
                   "prefill_tokens": 512,
                   "decode_tokens": 128,
               },
               "interval_generator_config": {
                   "type": "poisson",
                   "qps": 2.0,
               },
           },
           metrics_config={
               "output_dir": f"./results/{exp['model'].split('/')[-1]}"
           },
       )
       
       server_configs.append(server_config)
       benchmark_configs.append(benchmark_config)

   # Run experiments in parallel
   runner = ParallelBenchmarkRunner(max_workers=2)
   results = runner.run(
       list(zip(server_configs, benchmark_configs)), 
       benchmark_func=run_benchmark
   )

   print(f"Completed {len(results)} experiments")

Example 2: Sequential Execution with Resource Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For reproducible experiments or limited GPU resources:

.. code-block:: python

   from veeksha.orchestration import SequentialJobQueue
   from veeksha.orchestration.benchmark_orchestrator import managed_server
   from veeksha.config.server import ServerConfig
   from veeksha.config.benchmark import BenchmarkConfig
   from veeksha.benchmark import run_benchmark

   # Create job queue
   queue = SequentialJobQueue()

   # Define experiments (run sequentially to avoid resource conflicts)
   experiments = [
       ("meta-llama/Meta-Llama-3-8B-Instruct", 1, 8000),
       ("meta-llama/Meta-Llama-3-8B-Instruct", 2, 8001), 
       ("Qwen/Qwen2.5-0.5B-Instruct", 1, 8002),
   ]

   for model, tp_size, port in experiments:
       # Create server config
       server_config = ServerConfig(
           model=model,
           tensor_parallel_size=tp_size,
           port=port,
           auto_shutdown=True,
       )
       
       # Create benchmark config
       benchmark_config = BenchmarkConfig(
           max_completed_requests=100,
           timeout=600,
           client_config={"model": model},
           request_generator_config={
               "type": "synthetic",
               "length_generator_config": {
                   "type": "fixed",
                   "prefill_tokens": 512,
                   "decode_tokens": 128,
               },
               "interval_generator_config": {
                   "type": "poisson",
                   "qps": 1.0,
               },
           },
           metrics_config={
               "output_dir": f"./results/{model.split('/')[-1]}_tp{tp_size}"
           },
       )
       
       # Add to queue
       queue.add_job(server_config, benchmark_config, run_benchmark)

   # Execute all jobs sequentially
   results = queue.execute_all()

   print(f"Completed {len(results)} sequential experiments")
   for i, result in enumerate(results):
       if result:
           print(f"Experiment {i+1}: Success")
       else:
           print(f"Experiment {i+1}: Failed")

ServerConfig Options
--------------------

Basic Options
^^^^^^^^^^^^^

- :code:`engine`: Inference engine (:code:`"vllm"`, :code:`"vajra"`, etc.)
- :code:`model`: Model name or path
- :code:`host`: Server host address (default: :code:`"localhost"`)
- :code:`port`: Server port (default: :code:`8000`)
- :code:`api_key`: API authentication key

Resource Options
^^^^^^^^^^^^^^^^

- :code:`tensor_parallel_size`: Number of GPUs for tensor parallelism
- :code:`gpu_ids`: Specific GPU IDs to use (e.g., :code:`[0, 1, 3]`)
- :code:`dtype`: Model data type (:code:`"auto"`, :code:`"float16"`, :code:`"bfloat16"`)
- :code:`max_model_len`: Maximum context length

Lifecycle Options
^^^^^^^^^^^^^^^^^

- :code:`startup_timeout`: Seconds to wait for server startup (default: 300)
- :code:`health_check_interval`: Seconds between health checks (default: 2.0)
- :code:`auto_shutdown`: Automatically shutdown after benchmark (default: :code:`True`)

Additional Arguments
^^^^^^^^^^^^^^^^^^^^

Use :code:`additional_args` dict for engine-specific options:

.. code-block:: python

   server_config = ServerConfig(
       engine="vllm",
       model="meta-llama/Meta-Llama-3-8B-Instruct",
       additional_args={
           "rope_scaling": {"type": "dynamic", "factor": 2.0},
           "max_num_seqs": 256,
       }
   )

Use Cases
---------

Resource-Constrained Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you have limited GPUs and need to run many experiments:

.. code-block:: python

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

Manual Server Management
^^^^^^^^^^^^^^^^^^^^^^^^

If you need more control:

.. code-block:: python

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

Context Manager Pattern
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from veeksha.orchestration.vllm_server import VLLMServerManager

   with VLLMServerManager(server_config) as server:
       # Server is launched and ready
       run_benchmark(benchmark_config)
       # Server automatically shut down on exit

Extending to Other Inference Systems
------------------------------------

To add support for a new inference system:

1. Create a new manager class inheriting from :code:`BaseServerManager`
2. Implement :code:`_build_launch_command()` method
3. Optionally override :code:`health_check()` if needed

Example for a hypothetical system:

.. code-block:: python

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

Examples
--------

See :code:`veeksha/orchestration/examples/` for complete working examples:

- :code:`simple_example.py`: Basic single benchmark with server orchestration
- :code:`microbenchmark_orchestration.py`: Run prefill/decode probes with orchestration
- :code:`lmeval_orchestration.py`: Run lm_eval tasks with orchestration

Each example demonstrates:
- How to configure servers and workloads
- Error handling and best practices

Troubleshooting
---------------

Server fails to start
^^^^^^^^^^^^^^^^^^^^^

- Check GPU availability: :code:`nvidia-smi`
- Verify model access (HuggingFace login if needed)
- Increase :code:`startup_timeout` for large models
- Check server logs via :code:`server_manager.get_server_logs()`

Port conflicts
^^^^^^^^^^^^^^

- Use different ports for concurrent servers
- Check for existing processes: :code:`lsof -i :8000`

Resource issues
^^^^^^^^^^^^^^^

- Verify GPU memory with :code:`nvidia-smi`
- Adjust :code:`tensor_parallel_size` based on available GPUs
- Use :code:`gpu_ids` to explicitly assign GPUs

Future Enhancements
-------------------

Planned improvements (not yet implemented):

- [ ] Support for TGI, SGLang, and other inference systems
- [ ] Advanced resource scheduling for parallel experiments
- [ ] Persistent server pools
- [ ] Remote server management
- [ ] Integration with job schedulers (Slurm, etc.)

Related
-------

- Main benchmark documentation: :doc:`../../README`
- Capacity search: :doc:`../capacity_search/index`