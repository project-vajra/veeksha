"""
Base server manager for orchestrating LLM inference servers.

This module provides the abstract base class for managing the lifecycle
of LLM inference servers (launch, health check, shutdown).
"""

import abc
import subprocess
import time
from typing import Any, Dict, Optional

import requests

from veeksha.config.server import ServerConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


class BaseServerManager(abc.ABC):
    """Abstract base class for managing LLM inference servers.

    Subclasses should implement engine-specific launch commands and
    health check logic.
    """

    def __init__(self, config: ServerConfig):
        """Initialize the server manager.

        Args:
            config: Server configuration
        """
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._is_running = False
        self._log_file = None  # Store log file for cleanup

    @property
    def is_running(self) -> bool:
        """Check if server is currently running."""
        return (
            self._is_running
            and self.process is not None
            and self.process.poll() is None
        )

    @abc.abstractmethod
    def _build_launch_command(self) -> list[str]:
        """Build the command to launch the server.

        Returns:
            List of command arguments
        """

    def launch(self) -> bool:
        """Launch the inference server.

        Returns:
            True if launch was successful, False otherwise
        """
        if self.is_running:
            logger.warning(
                f"Server already running on {self.config.host}:{self.config.port}"
            )
            return True

        try:
            command = self._build_launch_command()
            logger.info(f"Launching server with command: {' '.join(command)}")

            # Set up environment variables
            import os

            env = os.environ.copy()

            # Set CUDA_VISIBLE_DEVICES if gpu_ids specified (but not for Vajra, which uses command-line args)
            gpu_env = self.config.get_gpu_env_var()
            if gpu_env is not None and self.config.engine.lower() != "vajra":
                env["CUDA_VISIBLE_DEVICES"] = gpu_env
                logger.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_env}")

            # Launch server process
            # Redirect output to a temporary file so we can check for errors
            import tempfile

            self._log_file = tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=".log", prefix="vllm_server_"
            )
            logger.info(f"Server logs: {self._log_file.name}")

            self.process = subprocess.Popen(
                command,
                env=env,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
            )

            logger.info(f"Server process started with PID: {self.process.pid}")
            self._is_running = True
            return True

        except Exception as e:
            logger.error(f"Failed to launch server: {e}")
            return False

    def health_check(self) -> bool:
        """Check if server is healthy and ready to accept requests.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            health_url = self.config.get_health_check_url()
            response = requests.get(health_url, timeout=5)

            # Most servers return 200 OK when healthy
            if response.status_code == 200:
                return True
            else:
                logger.debug(f"Health check failed with status: {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            logger.debug(f"Health check failed: {e}")
            return False

    def wait_for_ready(self, timeout: Optional[int] = None) -> bool:
        """Wait for server to become ready.

        Args:
            timeout: Maximum time to wait in seconds (uses config.startup_timeout if None)

        Returns:
            True if server became ready, False if timeout
        """
        if timeout is None:
            timeout = self.config.startup_timeout

        logger.info(f"Waiting for server to be ready (timeout: {timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if process is still alive
            if not self.is_running:
                logger.error("Server process terminated unexpectedly")
                # Read the log file to check for common errors
                if self._log_file:
                    try:
                        self._log_file.seek(0)
                        logs = self._log_file.read()

                        # Check for GPU memory error
                        if (
                            "Free memory on device" in logs
                            and "is less than desired GPU memory utilization" in logs
                        ):
                            import re

                            # Extract memory info from error message
                            match = re.search(
                                r"Free memory on device \(([0-9.]+)/([0-9.]+) GiB\).*desired GPU memory utilization.*\(([0-9.]+), ([0-9.]+) GiB\)",
                                logs,
                            )
                            if match:
                                free_mem, total_mem, util_frac, needed_mem = (
                                    match.groups()
                                )
                                logger.error(
                                    f"\n{'='*80}\n"
                                    f"GPU MEMORY ERROR: Insufficient GPU memory available\n"
                                    f"  Free memory:    {free_mem} GiB / {total_mem} GiB\n"
                                    f"  Required:       {needed_mem} GiB (utilization: {util_frac})\n"
                                    f"\n"
                                    f"Solutions:\n"
                                    f"  1. Free up GPU memory by stopping other processes\n"
                                    f"  2. Use a smaller model\n"
                                    f"{'='*80}"
                                )
                            else:
                                logger.error(
                                    "GPU memory error detected but couldn't parse details"
                                )
                        else:
                            # Show last 50 lines of logs for other errors
                            log_lines = logs.strip().split("\n")
                            recent_logs = "\n".join(log_lines[-50:])
                            logger.error(f"Recent server logs:\n{recent_logs}")
                    except Exception as e:
                        logger.error(f"Failed to read server logs: {e}")
                return False

            # Check health
            if self.health_check():
                elapsed = time.time() - start_time
                logger.info(f"Server is ready! (took {elapsed:.1f}s)")
                return True

            # Wait before next check
            time.sleep(self.config.health_check_interval)

        logger.error(f"Server did not become ready within {timeout}s")
        return False

    def shutdown(self, force: bool = False) -> bool:
        """Shutdown the server.

        Args:
            force: If True, force kill the process

        Returns:
            True if shutdown was successful, False otherwise
        """
        if not self.is_running:
            logger.warning("Server is not running")
            return True

        if self.process is None:
            logger.error("Server process is None, cannot shutdown")
            return False

        try:
            logger.info(f"Shutting down server (PID: {self.process.pid})")

            if force:
                self.process.kill()
                logger.info("Force killed server process")
            else:
                self.process.terminate()
                logger.info("Sent termination signal to server")

                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=30)
                    logger.info("Server shut down gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Server did not shut down gracefully, force killing")
                    self.process.kill()

            # Ensure process is reaped, ignore errors
            try:
                self.process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error waiting for process to exit: {e}")

            return True

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False
        finally:
            # Always reset state, even if exceptions occur
            self._is_running = False

            # Clean up log file
            if self._log_file:
                try:
                    self._log_file.close()
                    import os

                    os.unlink(self._log_file.name)
                    logger.debug(f"Removed log file: {self._log_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up log file: {e}")

    def get_server_logs(self, lines: int = 50) -> tuple[str, str]:
        """Get recent server logs.

        Args:
            lines: Number of lines to retrieve

        Returns:
            Tuple of (stdout, stderr)
        """
        if self.process is None:
            return "", ""

        # Note: This is a simple implementation that reads available output
        # For production, consider using proper log file management
        try:
            # Try to read without blocking
            pass

            stdout_lines = []
            stderr_lines = []

            # This is a simplified version - for production use proper logging
            return "", ""

        except Exception as e:
            logger.error(f"Error reading server logs: {e}")
            return "", ""

    def get_additional_args_dict(self) -> Dict[str, Any]:
        """Parse additional_args JSON string into a dictionary.

        Returns:
            Dictionary of parsed additional arguments
        """
        import json

        additional_args_dict: Dict[str, Any] = {}
        if self.config.additional_args:
            additional_args_dict = json.loads(self.config.additional_args)
        return additional_args_dict

    def __enter__(self):
        """Context manager entry."""
        self.launch()
        self.wait_for_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.config.auto_shutdown:
            self.shutdown()
