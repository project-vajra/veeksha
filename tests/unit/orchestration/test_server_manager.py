
import os
import time
import tempfile
from pathlib import Path

import pytest  # type: ignore[import]
from unittest.mock import MagicMock, patch, ANY
from veeksha.orchestration.server_manager import BaseServerManager
from veeksha.config.server import VllmServerConfig

pytestmark = pytest.mark.unit

class TestServerManager(BaseServerManager):
    """Concrete implementation for testing."""
    def _build_launch_command(self) -> list[str]:
        return ["python", "-m", "mock_server"]

@pytest.fixture
def server_config():
    return VllmServerConfig(
        host="localhost",
        port=8000,
        gpu_ids=[0],
        startup_timeout=1,
        health_check_interval=0.1
    )

@pytest.fixture
def manager(server_config):
    mgr = TestServerManager(server_config)
    mgr._is_port_in_use = MagicMock(return_value=False)
    return mgr

class TestBaseServerManager:
    
    @patch("subprocess.Popen")
    def test_launch_success(self, mock_popen, manager):
        """Test successful server launch."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        success, error = manager.launch()
        assert success
        assert error is None
        
        assert manager.is_running
        assert manager.process == mock_process
        mock_popen.assert_called_once()
        
        # Check environment variables
        call_args = mock_popen.call_args
        env = call_args[1]['env']
        assert env['CUDA_VISIBLE_DEVICES'] == "0"

    @patch("subprocess.Popen")
    def test_launch_already_running(self, mock_popen, manager):
        """Test launch when already running."""
        manager._is_running = True
        manager.process = MagicMock()
        manager.process.poll.return_value = None
        
        success, error = manager.launch()
        assert success
        assert error is None
        mock_popen.assert_not_called()

    @patch("subprocess.Popen")
    def test_launch_failure(self, mock_popen, manager):
        """Test launch failure."""
        mock_popen.side_effect = Exception("Launch failed")
        
        success, error = manager.launch()
        assert not success
        assert error == "Launch failed"
        assert not manager.is_running

    @patch("subprocess.Popen")
    def test_launch_fails_when_port_in_use(self, mock_popen, manager):
        """Ensure launch aborts when another process already uses the port."""
        manager._is_port_in_use.return_value = True

        success, error = manager.launch()

        assert not success
        assert "already in use" in error
        mock_popen.assert_not_called()

    @patch("requests.get")
    def test_health_check_success(self, mock_get, manager):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        assert manager.health_check()
        mock_get.assert_called_with("http://localhost:8000/health", timeout=5)

    @patch("requests.get")
    def test_health_check_failure_status(self, mock_get, manager):
        """Test health check failure due to status code."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response
        
        assert not manager.health_check()

    @patch("requests.get")
    def test_health_check_failure_exception(self, mock_get, manager):
        """Test health check failure due to exception."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Connection refused")
        
        assert not manager.health_check()

    @patch("time.sleep")
    @patch("requests.get")
    def test_wait_for_ready_success(self, mock_get, mock_sleep, manager):
        """Test wait_for_ready success."""
        # Simulate not ready then ready
        manager._is_running = True
        manager.process = MagicMock()
        manager.process.poll.return_value = None
        
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 503
        
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        
        mock_get.side_effect = [mock_response_fail, mock_response_success]
        
        assert manager.wait_for_ready(timeout=5)
        assert mock_get.call_count == 2

    @patch("time.sleep")
    @patch("requests.get")
    def test_wait_for_ready_timeout(self, mock_get, mock_sleep, manager):
        """Test wait_for_ready timeout."""
        manager._is_running = True
        manager.process = MagicMock()
        manager.process.poll.return_value = None
        
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response
        
        # Mock time.time to simulate timeout
        time_calls = {"value": 0}

        def time_mock():
            time_calls["value"] += 1
            return time_calls["value"] * 0.2  # 0.2, 0.4, 0.6, ...
        
        with patch("time.time", side_effect=time_mock):
            assert not manager.wait_for_ready(timeout=1)

    @patch("time.sleep")
    def test_wait_for_ready_process_died(self, mock_sleep, manager):
        """Test wait_for_ready when process dies."""
        manager._is_running = False # Process not running
        
        assert not manager.wait_for_ready(timeout=5)

    def test_shutdown(self, manager):
        """Test shutdown."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        manager.process = mock_process
        manager._is_running = True
        
        assert manager.shutdown()
        
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called()
        assert not manager.is_running

    def test_shutdown_force(self, manager):
        """Test forced shutdown."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        manager.process = mock_process
        manager._is_running = True
        
        assert manager.shutdown(force=True)
        
        mock_process.kill.assert_called_once()
        assert not manager.is_running

    def test_get_additional_args_dict(self):
        """Test parsing additional args."""
        # Test string (JSON)
        config = VllmServerConfig(additional_args='{"key": "value"}')
        manager = TestServerManager(config)
        args = manager.get_additional_args_dict()
        assert args == {"key": "value"}

        # Test None
        config = VllmServerConfig(additional_args=None)
        manager = TestServerManager(config)
        args = manager.get_additional_args_dict()
        assert args == {}

        # Test dict
        config = VllmServerConfig(additional_args={"key": "value"})
        manager = TestServerManager(config)
        args = manager.get_additional_args_dict()
        assert args == {"key": "value"}
        # Ensure it's a shallow copy
        args["new"] = "added"
        assert config.additional_args == {"key": "value"}

        # Test invalid JSON string
        config = VllmServerConfig(additional_args='{"invalid": json}')
        manager = TestServerManager(config)
        with pytest.raises(ValueError, match="Invalid JSON in additional_args"):
            manager.get_additional_args_dict()

        # Test invalid type - can't test directly since ServerConfig is frozen, but method handles it

    def test_auto_allocation(self):
        """Test auto-allocation of GPUs during launch."""
        # Config without explicit GPU IDs
        config = VllmServerConfig(gpu_ids=None, tensor_parallel_size=2)
        manager = TestServerManager(config)
        
        # Mock resource manager
        manager.resource_manager = MagicMock()
        manager.resource_manager.wait_for_resources.return_value = [("node1", 0), ("node1", 1)]
        
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process
            
            success, error = manager.launch()
            assert success
            assert error is None
            
            # Check that GPUs were allocated
            manager.resource_manager.wait_for_resources.assert_called_with(
                num_gpus=2, timeout=300, job_id=ANY, contiguous=True
            )
            
            # Check that config was updated
            assert manager.config.gpu_ids == [0, 1]
            
            # Check env var
            call_args = mock_popen.call_args
            env = call_args[1]['env']
            assert env['CUDA_VISIBLE_DEVICES'] == "0,1"

    def test_auto_allocation_non_contiguous(self):
        """Ensure we can request non-contiguous GPUs when flag is disabled."""
        config = VllmServerConfig(
            gpu_ids=None,
            tensor_parallel_size=2,
            require_contiguous_gpus=False,
        )
        manager = TestServerManager(config)

        manager.resource_manager = MagicMock()
        manager.resource_manager.wait_for_resources.return_value = [
            ("node1", 0),
            ("node2", 1),
        ]

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            success, error = manager.launch()
            assert success
            assert error is None

            manager.resource_manager.wait_for_resources.assert_called_with(
                num_gpus=2,
                timeout=300,
                job_id=ANY,
                contiguous=False,
            )

    def test_get_server_logs_reads_last_n_lines(self, tmp_path, manager):
        """Test that get_server_logs returns the last N lines from the log file."""
        import tempfile
        # Create a temporary file and write some lines
        tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log")
        try:
            tmp.write("line1\nline2\nline3\n")
            tmp.flush()
            # Attach the file object to the manager
            manager._log_file = tmp

            stdout, stderr = manager.get_server_logs(lines=2)
            assert stdout.strip() == "line2\nline3"
            assert stderr == ""
        finally:
            try:
                tmp.close()
            except Exception:
                pass

    def test_get_server_logs_no_log_file(self, manager):
        """If no log file was created, return two empty strings."""
        manager._log_file = None
        stdout, stderr = manager.get_server_logs(lines=10)
        assert stdout == ""
        assert stderr == ""

    @patch("subprocess.Popen")
    def test_logs_written_to_metrics_output_dir(
        self, mock_popen, tmp_path, monkeypatch
    ):
        """Server logs should live inside the benchmark output directory."""
        config = VllmServerConfig(
            host="localhost",
            port=8123,
            gpu_ids=[0],
            startup_timeout=1,
            health_check_interval=0.1,
        )
        manager = TestServerManager(config)
        monkeypatch.setenv("VEEKSHA_OUTPUT_DIR", str(tmp_path))

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 42
        mock_popen.return_value = mock_process

        success, _ = manager.launch()
        assert success

        log_path = Path(manager._log_file.name)
        assert log_path.parent == tmp_path
        assert log_path.exists()

        manager.shutdown()
        assert log_path.exists()

        monkeypatch.delenv("VEEKSHA_OUTPUT_DIR", raising=False)

