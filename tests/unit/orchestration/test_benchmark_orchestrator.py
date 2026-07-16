"""Tests for the current managed-engine orchestrator API."""

import os
from unittest.mock import MagicMock, patch

import pytest

from veeksha.config.endpoint import EndpointConfig
from veeksha.config.server import SglangServerConfig, VllmServerConfig
from veeksha.orchestration.benchmark_orchestrator import (
    create_server_manager,
    managed_server,
)
from veeksha.orchestration.managed_engines import (
    SglangOmniDockerRunner,
    VllmOmniDockerRunner,
)

pytestmark = pytest.mark.unit


def _vllm_config() -> VllmServerConfig:
    return VllmServerConfig(
        hf_model="meta/demo-model",
        deploy_config="/tmp/vllm-deploy.yaml",
    )


def _sglang_config() -> SglangServerConfig:
    return SglangServerConfig(
        model_path="meta/demo-model",
        deploy_config="/tmp/sglang-deploy.yaml",
        bootstrap="",
    )


class TestBenchmarkOrchestrator:
    def test_create_server_manager_vllm(self):
        manager = create_server_manager(_vllm_config(), output_dir="/tmp")
        assert isinstance(manager, VllmOmniDockerRunner)

    def test_create_server_manager_sglang(self):
        manager = create_server_manager(_sglang_config(), output_dir="/tmp")
        assert isinstance(manager, SglangOmniDockerRunner)

    @patch("veeksha.orchestration.benchmark_orchestrator.create_server_manager")
    def test_managed_server_context(self, mock_create):
        endpoint = EndpointConfig(
            engine_type="vllm",
            model="meta/demo-model",
            api_base="http://localhost:8000/v1",
            api_key="test-key",
            health_url="http://localhost:8000/v1/models",
            port=8000,
        )
        mock_manager = MagicMock()
        mock_manager.get_endpoint.return_value = endpoint
        mock_create.return_value = mock_manager

        with managed_server(_vllm_config(), output_dir="/tmp") as info:
            assert info["endpoint"] == endpoint
            assert info["api_base"] == endpoint.api_base
            assert info["api_key"] == endpoint.api_key
            assert info["server_manager"] == mock_manager
            mock_manager.start.assert_called_once()
            mock_manager.get_endpoint.assert_called_once()
            assert os.environ["OPENAI_API_KEY"] == "test-key"
            assert os.environ["OPENAI_API_BASE"] == "http://localhost:8000/v1"

        mock_manager.stop.assert_called_once()

    @patch("veeksha.orchestration.benchmark_orchestrator.create_server_manager")
    def test_managed_server_start_failure_still_stops(self, mock_create):
        mock_manager = MagicMock()
        mock_manager.start.side_effect = RuntimeError("failed to launch server")
        mock_create.return_value = mock_manager

        with pytest.raises(RuntimeError, match="failed to launch server"):
            with managed_server(_vllm_config(), output_dir="/tmp"):
                pass

        mock_manager.stop.assert_called_once()
