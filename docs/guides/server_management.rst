Server Management (ResourceManager & Server Manager)
=================================================

This guide documents Veeksha's resource manager and server manager helper classes
that are used to launch and manage LLM inference servers (for example, `vLLM`) while
coordinating GPU resources across experiments.

Overview
--------

- ResourceManager is responsible for detecting, tracking, and allocating GPU
  resources on one or more nodes in a cluster. It supports automatic GPU
  detection via pynvml and manual node registration.

- BaseServerManager provides a simple, reusable base class for launching,
  monitoring, and shutting down LLM inference servers. It integrates with
  ResourceManager to optionally auto-allocate GPUs when a server's config
  does not specify explicit GPU IDs.

ResourceManager
---------------

Key features:

- Auto-detect GPUs using `pynvml` (if installed).
- Manual node registration via `add_node()` for testing or custom clusters.
- Allocate contiguous GPUs on a single node (preferred) or multi-node
  allocations when requested.
- Wait for resources with `wait_for_resources()` with a timeout.
- Release resources by job id.

Important methods:

- `add_node(hostname: str, num_gpus: int, gpu_memory_mb: Optional[int] = None)`
- `allocate_resources(num_gpus: int, job_id: Optional[str] = None, contiguous: bool = True) -> Optional[ResourceMapping]`
- `wait_for_resources(num_gpus: int, timeout: Optional[float], job_id: Optional[str]) -> Optional[ResourceMapping]`
- `release_resources(job_id: str) -> bool`
- `get_resource_status()` returns a dict describing cluster usage.

Simple example — manual node and allocation

.. code-block:: python

    from veeksha.orchestration.resource_manager import ResourceManager

    rm = ResourceManager(detect_gpus=False)  # disable auto-detection for tests

    # Add a single node with 2 GPUs for testing/demo
    rm.add_node("node1", num_gpus=2, gpu_memory_mb=80_000)

    print(rm.get_resource_status())

    # Allocate 1 GPU
    mapping = rm.allocate_resources(1, job_id="job-test-1")
    assert mapping is not None, "Allocation failed"

    # mapping is a list of (hostname, gpu_id) pairs
    print("Allocated:", mapping)

    # Release
    success = rm.release_resources("job-test-1")
    assert success


Advanced: Wait for resources

.. code-block:: python

    mapping = rm.wait_for_resources(num_gpus=2, timeout=30, job_id="server_12345")
    if mapping is None:
        raise RuntimeError("Could not allocate GPUs in time")

    # Use mapping; when done call release_resources(job_id)


Server Manager (BaseServerManager)
----------------------------------

The `BaseServerManager` class encapsulates process lifecycle management for an
LLM inference server. Subclasses implement `_build_launch_command()` to return the
CLI necessary to start a server with the configuration defined in `ServerConfig`.

Base properties and behaviors:

- Uses `ServerConfig` for all server launch settings, including connection details,
  engine type, GPU selection behavior, and lifecycle options.
- If `ServerConfig.gpu_ids` is None, the manager attempts automatic allocation via
  `ResourceManager.wait_for_resources()` (auto-allocation happens during `launch()`).
- Launch uses subprocess, captures logs, and exposes them via `get_server_logs()`.
- Provides `health_check()` and `wait_for_ready()` to determine server readiness.
- On shutdown, the manager attempts graceful termination and releases any allocated
  GPU resources if present.

Example: Using `managed_server` for automatic lifecycle management

.. code-block:: python

  from veeksha.orchestration import managed_server
  from veeksha.config.server import ServerConfig

  # Configure server
  config = ServerConfig(
    engine="vllm",
    host="127.0.0.1",
    port=8000,
    api_key="test-key",
    gpu_ids=[0],  # or None to auto-allocate using ResourceManager
    tensor_parallel_size=1,
    startup_timeout=60,
    health_check_interval=1.0,
  )

  # Launch and manage server in a context
  with managed_server(config) as info:
    # Server is launched and ready inside the context
    print(f"Server ready at {info['api_base']}")
    # `info['server_manager']` exposes the underlying manager instance
    # and you can use the returned api_base/api_key to run workloads.

Context manager usage

.. code-block:: python

  # `managed_server` provides a convenient context manager: enter -> launch + wait
  from veeksha.orchestration import managed_server

  with managed_server(config) as info:
    # Server is launched and ready inside the context
    print(f"Server ready: {info['api_base']}")
    # Exiting context triggers shutdown if auto_shutdown is True

Auto GPU allocation and integration with ResourceManager
-------------------------------------------------------

When `ServerConfig.gpu_ids` is None, the server manager auto-allocates GPUs. Internally,
BaseServerManager will:

- Determine the number of GPUs required with `config.get_num_gpus()` (based on
  tensor_parallel_size or explicit gpu_ids).
- Call `ResourceManager.wait_for_resources()` with a job id that tracks the server
  allocation (for later release).
- Update the `ServerConfig` with the allocated `gpu_ids` so subsequent operations
  (e.g., environment variables) reflect the allocated GPUs.

For advanced customization you can replace the default `ResourceManager`
with your own implementation and register nodes via `add_node()` or use the
integration examples (`docs/examples/orchestration_example.py` and
`docs/examples/lmeval_orchestration.py`) for real-world patterns.

Practical tips and debugging
---------------------------

- View server logs with `BaseServerManager.get_server_logs()` to diagnose startup
  issues such as GPU memory errors or missing binaries.

- For GPU memory errors, BaseServerManager examines the server start log and logs
  helpful messages indicating free memory and suggestions to address insufficient
  GPU memory.

- The `ResourceManager` is thread-safe and uses an internal RLock for allocations,
  so it can be used across multiple manager instances in a single Python process for
  basic scheduling use-cases.

- For distributed multi-node pools, register each node via `add_node()` (or rely on
  detection if running on each host) and call `allocate_resources()` with
  contiguous=False if you permit non-contiguous multi-node allocations.

Examples & Further Reading
--------------------------

Two example scripts demonstrate practical usage of `managed_server` and
`ResourceManager` in realistic workflows (benchmarks and lm_eval):

- `docs/examples/orchestration_example.py` – shows automatic GPU allocation, launching
  servers using `managed_server`, and running benchmarks with `BenchmarkConfig`.
- `docs/examples/lmeval_orchestration.py` – demonstrates running `lm_eval` style tasks
  with automatic server management and multi-model comparisons.

See those scripts for runnable examples and a more complete end-to-end setup.

Conclusion
----------

Using `ResourceManager` together with `BaseServerManager` provides a stable and
flexible way to orchestrate GPU-backed inference servers across experiments.
The patterns above are used in the test suite to validate end-to-end behavior and
to support GPU-backed integration tests.
