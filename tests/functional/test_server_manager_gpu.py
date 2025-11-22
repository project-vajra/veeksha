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
from veeksha.orchestration.resource_manager import ResourceManager


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
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM not installed")


def _get_free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _env_path() -> str:
    # sys.executable is .../env/bin/python; we want the parent env path
    return os.path.dirname(os.path.dirname(sys.executable))


def _build_benchmark_config(output_dir: str, model: str, server_config: ServerConfig) -> BenchmarkConfig:
    return BenchmarkConfig(
        timeout=120,
        max_completed_requests=1,
        client_config=ClientConfig(model=model, tokenizer=model, min_tokens_param=None),
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


class TrackingResourceManager(ResourceManager):
    """Resource manager that records allocations for verification."""

    def __init__(self):
        super().__init__(detect_gpus=False)
        self.allocations = []
        self.releases = []
        self.hostname = socket.gethostname()
        self.add_node(self.hostname, num_gpus=1, gpu_memory_mb=80_000)

    def wait_for_resources(
        self,
        num_gpus,
        timeout=None,
        poll_interval=3.0,
        job_id=None,
        contiguous: bool = True,
    ):  # type: ignore[override]
        mapping = super().wait_for_resources(
            num_gpus=num_gpus,
            timeout=timeout,
            poll_interval=poll_interval,
            job_id=job_id,
            contiguous=contiguous,
        )
        if mapping and job_id:
            self.allocations.append((job_id, list(mapping)))
        return mapping

    def release_resources(self, job_id):  # type: ignore[override]
        success = super().release_resources(job_id)
        if success:
            self.releases.append(job_id)
        return success


class TestServerManagerGPU:
    @pytest.mark.gpu
    @pytest.mark.no_vllm_server
    def test_managed_server_runs_end_to_end(self, temp_output_dir: str, server_manager_model: str):
        """Launch a vLLM server via managed_server and run a tiny benchmark."""
        _require_gpu()
        _require_vllm()

        server_config = ServerConfig(
            engine="vllm",
            model=server_manager_model,
            host="127.0.0.1",
            port=_get_free_port(),
            api_key="gpu-integration-key",
            gpu_ids=[0],
            tensor_parallel_size=1,
            environment_path=_env_path(),
            startup_timeout=600,
            health_check_interval=1.0,
        )

        benchmark_config = _build_benchmark_config(temp_output_dir, server_manager_model, server_config)
        metrics = run_benchmark(benchmark_config)

        assert metrics.metric_store.num_completed_requests >= 1
        assert os.path.exists(temp_output_dir)

    @pytest.mark.gpu
    @pytest.mark.no_vllm_server
    def test_managed_server_auto_resource_allocation(self, temp_output_dir: str, server_manager_model: str, monkeypatch: pytest.MonkeyPatch):
        """Ensure ResourceManager-based GPU allocation works end-to-end."""
        _require_gpu()
        _require_vllm()

        tracker = TrackingResourceManager()
        monkeypatch.setattr(
            "veeksha.orchestration.server_manager.ResourceManager",
            lambda: tracker,
        )

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
        metrics = run_benchmark(benchmark_config)

        assert metrics.metric_store.num_completed_requests >= 1
        assert tracker.allocations, "Resource manager should allocate at least once"
        assert len(tracker.allocations) == len(tracker.releases)
        assert benchmark_config.server_config is not None
        assert benchmark_config.server_config.gpu_ids == [0]