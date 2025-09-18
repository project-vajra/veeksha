"""Shared fixtures for functional tests."""

import os
import subprocess
import tempfile
import time
import socket
import tempfile
from pathlib import Path
from typing import Iterator, Optional

import pytest
import requests

from veeksha.logger import init_logger

logger = init_logger("conftest")


class VLLMServer:
    """Helper class to manage vLLM server for testing."""

    def __init__(self, model: str, port: Optional[int] = None):
        self.model = model
        if port is None:            
            sock = socket.socket()
            sock.bind(('', 0))
            self.port = sock.getsockname()[1]
            sock.close()
        else:
            self.port = port
        self.base_url = f"http://localhost:{self.port}/v1"
        self.process = None
        logger.info(f"🚀 VLLMServer initialized with model: {model}, port: {self.port}")

    def start(self) -> None:
        """Start vLLM server."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.port),
            "--served-model-name", self.model,
            "--chat-template", '{% for message in messages %}{% if message["role"] == "user" %}User: {{ message["content"] }}\n{% elif message["role"] == "assistant" %}Assistant: {{ message["content"] }}\n{% endif %}{% endfor %}Assistant:',
        ]

        logger.info(f"🚀 Starting vLLM server with command: {' '.join(cmd)}")

        # Create a log file for the server
        self.log_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log')

        logger.info(f"📝 Server logs will be written to: {self.log_file.name}")

        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}  # Use first GPU
        )

        # Wait for server to be ready
        self._wait_for_server()

    def stop(self) -> None:
        """Stop vLLM server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            try:
                os.unlink(self.log_file.name)
            except OSError:
                pass

    def _wait_for_server(self, timeout: int = 180) -> None:
        """Wait for server to be ready."""
        start_time = time.time()
        last_log_time = start_time

        assert self.process is not None

        while time.time() - start_time < timeout:
            # Check if process has terminated
            if self.process.poll() is not None:
                logger.error(f"❌ ERROR: vLLM process terminated with code {self.process.returncode}")
                if hasattr(self, 'log_file') and self.log_file:
                    try:
                        self.log_file.seek(0)
                        logs = self.log_file.read()
                        if logs:
                            logger.error(f"Server logs:\n{logs}")
                    except Exception as e:
                        logger.error(f"Could not read logs: {e}")
                raise RuntimeError("vLLM server process terminated unexpectedly")

            # Print logs periodically
            current_time = time.time()
            if current_time - last_log_time > 5:  # Every 5 seconds
                logger.info(f"⏳ Still waiting for vLLM server... ({current_time - start_time:.1f}s elapsed)")

                # Print recent logs
                if hasattr(self, 'log_file') and self.log_file:
                    try:
                        self.log_file.seek(0)
                        logs = self.log_file.read()
                        if logs:
                            # Print last few lines
                            lines = logs.strip().split('\n')
                            recent_lines = lines[-3:] if len(lines) > 3 else lines
                            logger.info(f"Recent server logs:\n" + "\n".join(recent_lines))
                    except Exception as e:
                        logger.error(f"Could not read recent logs: {e}")

                last_log_time = current_time

            # Test server availability
            try:
                logger.info(f"🔍 Testing server at {self.base_url}/models...")
                response = requests.get(f"{self.base_url}/models", timeout=2)
                logger.info(f"📡 Response status: {response.status_code}")
                if response.status_code == 200:
                    logger.info(f"✅ vLLM server ready after {time.time() - start_time:.1f}s")
                    models = response.json()
                    model_ids = [m.get('id', 'unknown') for m in models.get('data', [])]
                    logger.info(f"Available models: {model_ids}")
                    return
                else:
                    logger.warning(f"⚠️ Server responded with status {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"🔌 Connection failed (expected during startup): {type(e).__name__}")

            time.sleep(2)

        # Final log dump if timeout
        logger.error("❌ Timeout reached")
        if hasattr(self, 'log_file') and self.log_file:
            try:
                self.log_file.seek(0)
                logs = self.log_file.read()
                if logs:
                    logger.error(f"Server logs:\n{logs}")
                else:
                    logger.error("No server logs available")
            except Exception as e:
                logger.error(f"Could not read final logs: {e}")

        raise RuntimeError(f"vLLM server failed to start within {timeout} seconds")


@pytest.fixture(scope="session")
def test_model() -> str:
    """Model to use for testing."""
    # Use a small model for faster testing
    # Options: facebook/opt-125m, gpt2, microsoft/phi-2
    model = os.environ.get("TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    logger.info(f"🎯 test_model fixture returning: {model}")
    return model


@pytest.fixture(scope="session")
def vllm_server(test_model: str) -> Iterator[VLLMServer]:
    """Start vLLM server for testing."""
    logger.info(f"🔧 vllm_server fixture called with model: {test_model}")
    logger.info(f"Starting vLLM server for model: {test_model}")
    logger.info(f"🏗️ Creating VLLMServer instance...")
    server = VLLMServer(test_model)
    logger.info(f"✅ VLLMServer instance created, starting server...")
    try:
        server.start()
        logger.info(f"🎉 vLLM server started successfully!")
        yield server
    finally:
        logger.info(f"🧹 Stopping vLLM server...")
        server.stop()
        logger.info(f"✅ vLLM server stopped")


@pytest.fixture
def temp_output_dir() -> Iterator[str]:
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_trace_file() -> str:
    """Path to sample trace file."""
    trace_path = Path(__file__).parent.parent.parent / "veeksha" / "data" / "processed_traces" / "sharegpt_8k_filtered_stats_llama2_tokenizer.csv"
    if not trace_path.exists():
        pytest.skip(f"Sample trace file not found: {trace_path}")
    return str(trace_path)


@pytest.fixture(autouse=True)
def setup_test_env(request) -> Iterator[None]:
    """Set environment variables automatically for all functional tests."""
    logger.info(f"🔧 setup_test_env called for test: {request.node.name}")
    old_env = {}
    new_env = {}

    # Only set vLLM env vars if the test needs it and server is available
    if "gpu" in [mark.name for mark in request.node.iter_markers()]:
        logger.info("Test has 'gpu' marker, setting up vLLM environment...")
        try:
            logger.info(f"🔍 Getting vllm_server fixture...")
            vllm_server = request.getfixturevalue("vllm_server")
            logger.info(f"✅ Got vllm_server fixture, updating env vars...")
            new_env.update({
                "OPENAI_API_KEY": "",
                "OPENAI_API_BASE": vllm_server.base_url,
            })
        except Exception as e:
            # vLLM not available, skip if GPU test
            logger.error(f"❌ Failed to get vLLM server: {e}")
            pytest.skip("vLLM not available for GPU test")

    # Save old values
    for key in new_env:
        old_env[key] = os.environ.get(key)
        os.environ[key] = new_env[key]

    try:
        yield
    finally:
        # Restore old values
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
