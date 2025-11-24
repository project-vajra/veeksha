"""GPU-backed functional tests for server manager orchestration."""

import os
import shutil
import socket
import sys
import importlib.util

import pytest

from veeksha.benchmark import run_benchmark
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import ClientConfig
from veeksha.config.generators.interval_generator.poisson_generator import (
    PoissonRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.length_generator.fixed_generator import (
    FixedRequestLengthGeneratorConfig,
)
from veeksha.config.generators.request_generator.synthetic_generator import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.config.metrics import MetricsConfig
from veeksha.config.server import ServerConfig
from veeksha.orchestration import managed_server
# Use the production ResourceManager via the server manager internals in tests


@pytest.fixture(scope="module")
def server_manager_model() -> str:
    return os.environ.get("SERVER_MANAGER_TEST_MODEL", "facebook/opt-125m")

pytestmark = pytest.mark.functional


def _require_gpu():
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU not available for server manager tests")
    except Exception:
        if shutil.which("nvidia-smi") is None:
            pytest.skip("nvidia-smi not found; skipping GPU server manager tests")


def _require_vllm():
    # If VLLM_PYTHON is set, we assume the environment has vllm installed
    if os.environ.get("VLLM_PYTHON"):
        return

    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM not installed")


def _get_free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _env_path() -> str:
    # If VLLM_PYTHON is set, use that environment
    vllm_python = os.environ.get("VLLM_PYTHON")
    if vllm_python:
        # vllm_python is .../env/bin/python; we want the parent env path
        return os.path.dirname(os.path.dirname(vllm_python))
    
    # sys.executable is .../env/bin/python; we want the parent env path
    return os.path.dirname(os.path.dirname(sys.executable))


def _build_benchmark_config(output_dir: str, model: str, server_config: ServerConfig) -> BenchmarkConfig:
    return BenchmarkConfig(
        timeout=120,
        max_completed_requests=1,
        # Use completions API for synthetic text prompts to avoid vLLM chat
        # template issues for models/tokenizers that don't define chat templates
        client_config=ClientConfig(
            model=model,
            tokenizer=model,
            min_tokens_param=None,
            llm_api="openai_completions",
            address_append_value="completions",
        ),
        metrics_config=MetricsConfig(
            output_dir=output_dir,
            stream_metrics=False,
            should_write_metrics_to_wandb=False,
        ),
        request_generator_config=SyntheticRequestGeneratorConfig(
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=0.1),
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=8,
                decode_tokens=4,
            ),
        ),
        server_config=server_config,
    )


# Note: This test suite is intended to run against a real GPU resource
# and should therefore not attempt to mock ResourceManager behavior. The
# tests below rely on the production `ResourceManager` implementation.


class TestServerManagerGPU:
    @pytest.mark.gpu
    @pytest.mark.no_vllm_server
    def test_managed_server_auto_resource_allocation(self, temp_output_dir: str, server_manager_model: str):
        """Ensure ResourceManager-based GPU allocation works end-to-end."""
        _require_gpu()
        _require_vllm()


        server_config = ServerConfig(
            engine="vllm",
            model=server_manager_model,
            host="127.0.0.1",
            port=_get_free_port(),
            api_key="gpu-resource-key",
            gpu_ids=None,  # trigger auto-allocation path
            tensor_parallel_size=1,
            environment_path=_env_path(),
            startup_timeout=600,
            health_check_interval=1.0,
        )

        benchmark_config = _build_benchmark_config(temp_output_dir, server_manager_model, server_config)

        server_manager = None
        with managed_server(server_config) as server_info:
            metrics = run_benchmark(benchmark_config)

            # Capture server_manager for later inspection after shutdown
            server_manager = server_info["server_manager"]

            # Print server logs for debugging / visibility
            stdout, stderr = server_manager.get_server_logs(lines=200)
            print("\n=== Server Logs (stdout) ===")
            print(stdout)
            if stderr:
                print("\n=== Server Logs (stderr) ===")
                print(stderr)

            # Verify that the server manager allocated GPUs using the real
            # ResourceManager implementation. We expect at least one job_id to
            # be present in allocated_resources while the server is running.
            allocated = server_manager.resource_manager.allocated_resources
            assert allocated, "Resource manager should have allocated GPUs for the server"

        # Print server logs again after server has shut down, if available
        if server_manager is not None:
            stdout, stderr = server_manager.get_server_logs(lines=200)
            print("\n=== Server Logs (stdout) after shutdown ===")
            print(stdout)
            if stderr:
                print("\n=== Server Logs (stderr) after shutdown ===")
                print(stderr)

        assert metrics.metric_store.num_completed_requests >= 1
        assert benchmark_config.server_config is not None