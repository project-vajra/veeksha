"""Integration tests covering managed server orchestration end-to-end."""

from __future__ import annotations

import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

import pytest  # type: ignore[import]
import requests

from veeksha.capacity_search.capacity_search import CapacitySearch
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.capacity_search import CapacitySearchConfig
from veeksha.config.client import ClientConfig
from veeksha.config.generators.interval_generator.static_generator import (
    StaticRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.length_generator.fixed_generator import (
    FixedRequestLengthGeneratorConfig,
)
from veeksha.config.generators.request_generator.synthetic_generator import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.config.metrics import MetricsConfig
from veeksha.config.server import ServerConfig
from veeksha.config.slo import ConstantSloConfig
from veeksha.orchestration.benchmark_orchestrator import managed_server
from veeksha.orchestration.resource_manager import ResourceManager, ResourceMapping
from veeksha.orchestration.vllm_server import VLLMServerManager

DUMMY_SERVER = (
    Path(__file__).resolve().parents[2]
    / "helpers"
    / "dummy_llm_server.py"
)

pytestmark = pytest.mark.functional


class TrackingResourceManager(ResourceManager):
    """Resource manager that records allocations/releases for assertions."""

    def __init__(self):
        super().__init__(detect_gpus=False)
        self.add_node("tracker-node", num_gpus=4, gpu_memory_mb=24_000)
        self.allocations: List[Tuple[str, List[Tuple[str, int]]]] = []
        self.releases: List[str] = []

    def allocate_resources(
        self,
        num_gpus: int,
        job_id: Optional[str] = None,
        contiguous: bool = True,
    ) -> Optional[ResourceMapping]:
        mapping = super().allocate_resources(num_gpus, job_id=job_id, contiguous=contiguous)
        if mapping and job_id:
            self.allocations.append((job_id, list(mapping)))
        return mapping

    def release_resources(self, job_id: str) -> bool:  # type: ignore[override]
        released = super().release_resources(job_id)
        if released:
            self.releases.append(job_id)
        return released


@pytest.fixture()
def tracking_resource_manager(monkeypatch) -> TrackingResourceManager:
    tracker = TrackingResourceManager()
    monkeypatch.setattr(
        "veeksha.orchestration.server_manager.ResourceManager",
        lambda: tracker,
    )
    return tracker


@pytest.fixture()
def dummy_server_launcher(monkeypatch):
    def _build_command(self):
        return [
            sys.executable,
            str(DUMMY_SERVER),
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--model",
            self.config.model,
        ]

    monkeypatch.setattr(
        VLLMServerManager,
        "_build_launch_command",
        _build_command,
    )


def _get_free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class DummyRequestLevelMetrics:
    def __init__(self, metrics: Dict[str, List[float]]):
        self._metrics = metrics

    def to_dict(self):
        return self._metrics


class DummyServiceMetrics:
    def __init__(self, metrics: Dict[str, List[float]]) -> None:
        self.metric_store = SimpleNamespace(
            request_level_metrics=DummyRequestLevelMetrics(metrics)
        )
        self.output_dir = ""


@pytest.fixture()
def stub_benchmark(monkeypatch):
    def _fake_run_benchmark(benchmark_config: BenchmarkConfig):
        qps_dir = Path(benchmark_config.metrics_config.output_dir).name
        qps = float(qps_dir)
        if qps <= 1.0:
            metrics = {"ttft": [0.2, 0.3], "tbt": [[0.05, 0.04]]}
        else:
            metrics = {"ttft": [0.8, 0.9], "tbt": [[0.2, 0.25]]}
        return DummyServiceMetrics(metrics)

    monkeypatch.setattr(
        "veeksha.capacity_search.benchmark_wrapper.run_benchmark_wrapped",
        _fake_run_benchmark,
    )


def _build_managed_server_config() -> ServerConfig:
    return ServerConfig(
        engine="vllm",
        model="dummy/model",
        host="127.0.0.1",
        port=_get_free_port(),
        tensor_parallel_size=2,
        gpu_ids=None,
        startup_timeout=10,
        health_check_interval=0.1,
    )


def test_managed_server_e2e_launches_and_releases(
    tracking_resource_manager, dummy_server_launcher
):
    config = _build_managed_server_config()

    with managed_server(config) as info:
        response = requests.get(f"{info['api_base']}/models", timeout=5)
        assert response.status_code == 200
        assert len(tracking_resource_manager.allocations) == 1

    # wait briefly for shutdown to release port/resources
    time.sleep(0.5)
    assert len(tracking_resource_manager.allocations) == 1
    assert len(tracking_resource_manager.releases) == 1
    assert tracking_resource_manager.get_free_gpus() == tracking_resource_manager.get_total_gpus()


def test_capacity_search_server_per_run_restarts_servers(
    tmp_path,
    tracking_resource_manager,
    dummy_server_launcher,
    stub_benchmark,
):
    metrics_dir = tmp_path / "bench_metrics"
    capsearch_dir = tmp_path / "capsearch"

    server_config = _build_managed_server_config()

    benchmark_config = BenchmarkConfig(
        max_completed_requests=1,
        timeout=5,
        client_config=ClientConfig(model="dummy/model", tokenizer="dummy/model"),
        metrics_config=MetricsConfig(
            output_dir=str(metrics_dir),
            stream_metrics=False,
            should_write_metrics_to_wandb=False,
        ),
        request_generator_config=SyntheticRequestGeneratorConfig(
            interval_generator_config=StaticRequestIntervalGeneratorConfig(),
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=8,
                decode_tokens=4,
            ),
        ),
        server_config=server_config,
    )

    capacity_config = CapacitySearchConfig(
        start_qps=1.0,
        max_iterations=2,
        min_search_granularity=0.1,
        output_dir=str(capsearch_dir),
        enable_experiment_cache=False,
        benchmark_config=benchmark_config,
        slos=[ConstantSloConfig(metric="ttft", value=0.5, percentile=0.5)],
        server_per_qps_run=True,
    )

    result = CapacitySearch(capacity_config).search()

    assert result.get("max_qps_under_sla") == pytest.approx(1.0)
    assert len(tracking_resource_manager.allocations) >= 2
    assert len(tracking_resource_manager.allocations) == len(tracking_resource_manager.releases)
    assert tracking_resource_manager.get_free_gpus() == tracking_resource_manager.get_total_gpus()
