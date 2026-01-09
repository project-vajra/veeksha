
import pytest  # type: ignore[import]
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit
from veeksha.orchestration.benchmark_orchestrator import create_server_manager, managed_server
from veeksha.config.server import VllmServerConfig, VajraServerConfig, SglangServerConfig, BaseServerConfig
from veeksha.orchestration.vllm_server import VLLMServerManager
from veeksha.orchestration.vajra_server import VajraServerManager
from veeksha.orchestration.sglang_server import SGLangServerManager

class TestBenchmarkOrchestrator:
    
    def test_create_server_manager_vllm(self):
        config = VllmServerConfig()
        manager = create_server_manager(config, output_dir="/tmp")
        assert isinstance(manager, VLLMServerManager)

    def test_create_server_manager_vajra(self):
        config = VajraServerConfig()
        manager = create_server_manager(config, output_dir="/tmp")
        assert isinstance(manager, VajraServerManager)

    def test_create_server_manager_sglang(self):
        config = SglangServerConfig()
        manager = create_server_manager(config, output_dir="/tmp")
        assert isinstance(manager, SGLangServerManager)

    @patch("veeksha.orchestration.benchmark_orchestrator.create_server_manager")
    def test_managed_server_context(self, mock_create):
        """Test the managed_server context manager."""
        config = VllmServerConfig(
            host="localhost",
            port=8000,
            api_key="test-key",
        )
        
        mock_manager = MagicMock()
        mock_manager.launch.return_value = True
        mock_manager.wait_for_ready.return_value = True
        mock_create.return_value = mock_manager
        
        with managed_server(config, output_dir="/tmp") as info:
            assert info["api_base"] == "http://localhost:8000/v1"
            assert info["api_key"] == "test-key"
            assert info["server_manager"] == mock_manager
            
            mock_manager.launch.assert_called_once()
            mock_manager.wait_for_ready.assert_called_once()
            
            # Check env vars
            import os
            assert os.environ["OPENAI_API_KEY"] == "test-key"
            assert os.environ["OPENAI_API_BASE"] == "http://localhost:8000/v1"
        
        mock_manager.shutdown.assert_called_once()

    @patch("veeksha.orchestration.benchmark_orchestrator.create_server_manager")
    def test_managed_server_launch_failure(self, mock_create):
        """Test managed_server when launch fails."""
        config = VllmServerConfig()
        mock_manager = MagicMock()
        mock_manager.launch.return_value = False
        mock_create.return_value = mock_manager
        
        with pytest.raises(RuntimeError, match="Failed to launch server"):
            with managed_server(config, output_dir="/tmp"):
                pass
        
        mock_manager.shutdown.assert_called_once()

    @patch("veeksha.orchestration.benchmark_orchestrator.create_server_manager")
    def test_managed_server_ready_failure(self, mock_create):
        """Test managed_server when wait_for_ready fails."""
        config = VllmServerConfig()
        mock_manager = MagicMock()
        mock_manager.launch.return_value = True
        mock_manager.wait_for_ready.return_value = False
        mock_create.return_value = mock_manager
        
        with pytest.raises(RuntimeError, match="Server failed to become ready"):
            with managed_server(config, output_dir="/tmp"):
                pass
                
        mock_manager.shutdown.assert_called_once()
