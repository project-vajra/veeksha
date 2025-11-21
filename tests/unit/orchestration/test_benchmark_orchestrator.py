
import pytest
from unittest.mock import MagicMock, patch
from veeksha.orchestration.benchmark_orchestrator import create_server_manager, managed_server
from veeksha.config.server import ServerConfig
from veeksha.orchestration.vllm_server import VLLMServerManager
from veeksha.orchestration.vajra_server import VajraServerManager
from veeksha.orchestration.sglang_server import SGLangServerManager

class TestBenchmarkOrchestrator:
    
    def test_create_server_manager_vllm(self):
        config = ServerConfig(engine="vllm")
        manager = create_server_manager(config)
        assert isinstance(manager, VLLMServerManager)

    def test_create_server_manager_vajra(self):
        config = ServerConfig(engine="vajra")
        manager = create_server_manager(config)
        assert isinstance(manager, VajraServerManager)

    def test_create_server_manager_sglang(self):
        config = ServerConfig(engine="sglang")
        manager = create_server_manager(config)
        assert isinstance(manager, SGLangServerManager)

    def test_create_server_manager_invalid(self):
        config = ServerConfig(engine="invalid")
        with pytest.raises(ValueError, match="Unsupported engine"):
            create_server_manager(config)

    @patch("veeksha.orchestration.benchmark_orchestrator.create_server_manager")
    def test_managed_server_context(self, mock_create):
        """Test the managed_server context manager."""
        config = ServerConfig(
            engine="vllm",
            host="localhost",
            port=8000,
            api_key="test-key",
            auto_shutdown=True
        )
        
        mock_manager = MagicMock()
        mock_manager.launch.return_value = True
        mock_manager.wait_for_ready.return_value = True
        mock_create.return_value = mock_manager
        
        with managed_server(config) as info:
            assert info["api_base"] == "http://localhost:8000/v1"
            assert info["api_key"] == "test-key"
            assert info["server_manager"] == mock_manager
            
            mock_manager.launch.assert_called_once()
            mock_manager.wait_for_ready.assert_called_once()
            
            # Check env vars
            import os
            assert os.environ["OPENAI_API_KEY"] == "test-key"
            assert os.environ["OPENAI_API_BASE"] == "http://localhost:8000/v1"
        
        # Verify shutdown called on exit
        mock_manager.shutdown.assert_called_once()

    @patch("veeksha.orchestration.benchmark_orchestrator.create_server_manager")
    def test_managed_server_launch_failure(self, mock_create):
        """Test managed_server when launch fails."""
        config = ServerConfig(engine="vllm")
        mock_manager = MagicMock()
        mock_manager.launch.return_value = False
        mock_create.return_value = mock_manager
        
        with pytest.raises(RuntimeError, match="Failed to launch server"):
            with managed_server(config):
                pass
        
        # Shutdown should still be called in finally block if auto_shutdown is True
        # But wait, if launch fails, we raise RuntimeError inside the try block.
        # The finally block will execute.
        mock_manager.shutdown.assert_called_once()

    @patch("veeksha.orchestration.benchmark_orchestrator.create_server_manager")
    def test_managed_server_ready_failure(self, mock_create):
        """Test managed_server when wait_for_ready fails."""
        config = ServerConfig(engine="vllm")
        mock_manager = MagicMock()
        mock_manager.launch.return_value = True
        mock_manager.wait_for_ready.return_value = False
        mock_create.return_value = mock_manager
        
        with pytest.raises(RuntimeError, match="Server failed to become ready"):
            with managed_server(config):
                pass
                
        mock_manager.shutdown.assert_called_once()

    @patch("veeksha.orchestration.benchmark_orchestrator.create_server_manager")
    def test_managed_server_no_auto_shutdown(self, mock_create):
        """Test managed_server with auto_shutdown=False."""
        config = ServerConfig(engine="vllm", auto_shutdown=False)
        mock_manager = MagicMock()
        mock_manager.launch.return_value = True
        mock_manager.wait_for_ready.return_value = True
        mock_create.return_value = mock_manager
        
        with managed_server(config):
            pass
            
        mock_manager.shutdown.assert_not_called()
