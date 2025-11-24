"""
Base server manager for orchestrating LLM inference servers.

This module provides the abstract base class for managing the lifecycle
of LLM inference servers (launch, health check, shutdown).
"""

import abc
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import IO, Any, Dict, Optional

import requests

from veeksha.config.server import ServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration.resource_manager import ResourceManager

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
        self._log_file_path: Optional[Path] = None
        self._delete_log_file_on_cleanup = True
        self.resource_manager = ResourceManager()
        self._allocated_job_id: Optional[str] = None  # Track allocated resources

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

    def _create_log_file(self) -> IO[str]:
        """Create a log file for the server process."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = (
            f"server_logs_{self.config.engine.lower()}_{self.config.host}_"
            f"{self.config.port}_{timestamp}.log"
        )

        output_dir = os.environ.get("VEEKSHA_OUTPUT_DIR")
        if output_dir:
            log_dir = Path(output_dir)
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / log_filename
                log_file = open(log_path, "w+", encoding="utf-8")
                self._log_file_path = log_path
                self._delete_log_file_on_cleanup = False
                return log_file
            except Exception as exc:  # pragma: no cover - fallback path
                logger.warning(
                    "Unable to create server log file in benchmark output directory "
                    f"'{output_dir}': {exc}. Falling back to a temporary file."
                )

        temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".log", prefix="llm_server_"
        )
        self._log_file_path = Path(temp_file.name)
        self._delete_log_file_on_cleanup = True
        return temp_file

    def launch(self) -> tuple[bool, Optional[str]]:
        """Launch the inference server.

        Returns:
            Tuple of (True if launch was successful, False otherwise, error message if any)
        """
        if self.is_running:
            logger.warning(
                f"Server already running on {self.config.host}:{self.config.port}"
            )
            return True, None

        if self._is_port_in_use():
            error_msg = (
                f"Port {self.config.port} on host '{self.config.host}' is already in use. "
                "Stop the existing process or update server_config.port to a free port."
            )
            logger.error(error_msg)
            return False, error_msg

        try:
            # Auto-allocate GPUs if not specified
            if self.config.gpu_ids is None:
                num_gpus = self.config.get_num_gpus()
                logger.info(f"Auto-allocating {num_gpus} GPUs for server...")

                job_id = (
                    f"server_{self.config.host}_{self.config.port}_{int(time.time())}"
                )
                resource_mapping = self.resource_manager.wait_for_resources(
                    num_gpus=num_gpus,
                    timeout=300,  # 5 minute timeout
                    job_id=job_id,
                    contiguous=self.config.require_contiguous_gpus,
                )

                if resource_mapping is None:
                    logger.error(f"Failed to allocate {num_gpus} GPUs for server")
                    return False, f"Failed to allocate {num_gpus} GPUs for server"

                # Track allocated job id for later release / Vajra mapping
                self._allocated_job_id = job_id

                # Extract GPU IDs from resource mapping
                gpu_ids = [gpu_id for _, gpu_id in resource_mapping]

                # Update config with allocated GPUs
                # Create a new config object with the allocated gpu_ids
                from dataclasses import replace

                self.config = replace(self.config, gpu_ids=gpu_ids)

                logger.info(f"Allocated GPUs {gpu_ids} for server")

            command = self._build_launch_command()
            logger.info(f"Launching server with command: {' '.join(command)}")

            # Set up environment variables
            env = os.environ.copy()

            # If an environment path is provided in the config, prepend its
            # bin/Scripts directory to PATH so the subprocess resolves the
            # `python` executable from that environment.
            env_path = getattr(self.config, "environment_path", None)
            if env_path:
                # Determine platform-specific scripts directory
                scripts_dir = "Scripts" if os.name == "nt" else "bin"
                bin_dir = os.path.join(env_path, scripts_dir)
                if os.path.isdir(bin_dir):
                    old_path = env.get("PATH", "")
                    env["PATH"] = f"{bin_dir}{os.pathsep}{old_path}"
                    logger.info(f"Prepended {bin_dir} to PATH for subprocess")
                else:
                    raise ValueError(
                        f"Configured environment_path '{env_path}' does not contain {scripts_dir} at {bin_dir}"
                    )

            # Set CUDA_VISIBLE_DEVICES if gpu_ids specified
            gpu_env = self.config.get_gpu_env_var()
            if gpu_env is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_env
                logger.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_env}")

            # Launch server process
            # Redirect output to a log file inside the benchmark output directory
            self._log_file = self._create_log_file()

            self.process = subprocess.Popen(
                command,
                env=env,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
            )

            logger.info(f"Server process started with PID: {self.process.pid}")
            self._is_running = True
            return True, None

        except Exception as e:
            # If we allocated GPUs earlier, make sure to release them
            if self._allocated_job_id is not None:
                logger.info(
                    f"Releasing allocated resources for job {self._allocated_job_id} due to launch failure"
                )
                self.resource_manager.release_resources(self._allocated_job_id)
                self._allocated_job_id = None
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            if self._delete_log_file_on_cleanup and self._log_file_path is not None:
                if self._log_file_path.exists():
                    self._log_file_path.unlink()
                self._log_file_path = None
                self._delete_log_file_on_cleanup = True
            return False, str(e)

    def _is_port_in_use(self) -> bool:
        """Return True if the configured host:port already has an active listener."""
        host = self.config.host
        port = self.config.port

        try:
            addr_info = socket.getaddrinfo(
                host,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            logger.debug(
                "Skipping port availability check because host '%s' cannot be resolved: %s",
                host,
                exc,
            )
            return False

        for family, socktype, proto, _, sockaddr in addr_info:
            try:
                with socket.socket(family, socktype, proto) as sock:
                    sock.settimeout(1.0)
                    if sock.connect_ex(sockaddr) == 0:
                        return True
            except OSError as exc:
                logger.debug(
                    "Port availability probe failed for %s:%s with %s",
                    host,
                    port,
                    exc,
                )
                continue

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
        success = True
        try:
            if not self.is_running:
                logger.warning("Server is not running")
            elif self.process is None:
                logger.error("Server process is None, cannot shutdown")
                success = False
            else:
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
                        logger.warning(
                            "Server did not shut down gracefully, force killing"
                        )
                        self.process.kill()

                # Ensure process is reaped, ignore errors
                try:
                    self.process.wait(timeout=5)
                except Exception as e:
                    logger.warning(f"Error waiting for process to exit: {e}")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            success = False
        finally:
            # Always reset state and clean up resources, even if exceptions occur
            self._is_running = False

            # Release allocated resources if any
            if self._allocated_job_id is not None:
                try:
                    logger.info(
                        f"Releasing allocated resources for job {self._allocated_job_id}"
                    )
                    self.resource_manager.release_resources(self._allocated_job_id)
                except Exception as e:
                    logger.error(f"Error releasing resources: {e}")
                finally:
                    self._allocated_job_id = None

            # Clean up log file
            if self._log_file:
                try:
                    self._log_file.close()
                except Exception as e:
                    logger.warning(f"Failed to close log file: {e}")
                finally:
                    self._log_file = None

            if (
                self._delete_log_file_on_cleanup
                and self._log_file_path is not None
                and self._log_file_path.exists()
            ):
                try:
                    os.unlink(self._log_file_path)
                    logger.debug(f"Removed log file: {self._log_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up log file: {e}")

            self._log_file_path = None
            self._delete_log_file_on_cleanup = True

        return success

    def get_server_logs(self, lines: int = 50) -> tuple[str, str]:
        """Get recent server logs.

        Args:
            lines: Number of lines to retrieve

        Returns:
            Tuple of (stdout, stderr). Note that by default the server
            subprocess redirects stderr into stdout, so stderr will usually
            be an empty string and stdout will contain both streams.
        """
        # If we never set up a log file we can't return anything useful
        log_path: Optional[Path] = None
        if self._log_file_path is not None:
            log_path = self._log_file_path
        elif self._log_file is not None:
            log_path = Path(self._log_file.name)
        else:
            return "", ""

        # Note: This is a simple implementation that reads available output
        # For production, consider using proper log file management (rotation,
        # streaming, or structured logs). The server's launch() redirects
        # both stdout and stderr to the same temporary file, so we return
        # that combined stream as stdout and leave stderr empty.
        try:
            # Ensure any buffered output is flushed before we read the file
            if self._log_file is not None:
                try:
                    self._log_file.flush()
                except Exception:
                    # Ignore any flush errors; we'll still attempt to read the file
                    pass

            if not log_path.exists():
                return "", ""

            # Read the log file content from disk rather than relying on the
            # file object's current pointer. This avoids disturbing the file
            # pointer used by the subprocess and reads bytes safely even while
            # the subprocess is still running.
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.read().splitlines()

            if lines <= 0:
                tail = "\n".join(all_lines)
            else:
                tail = "\n".join(all_lines[-lines:])

            # stderr is merged into stdout by launch(); return stderr as empty.
            return tail, ""
        except Exception as e:
            logger.exception(f"Error reading server logs: {e}")
            return "", ""

    def get_additional_args_dict(self) -> Dict[str, Any]:
        """Parse additional_args into a dictionary.

        additional_args can be None, a dict, or a JSON string.
        - If None, returns an empty dict.
        - If already a dict, returns a shallow copy.
        - If a str, attempts to parse as JSON; raises ValueError on invalid JSON.
        - For any other type, raises TypeError.

        Returns:
            Dictionary of parsed additional arguments
        """
        import copy
        import json

        additional_args = self.config.additional_args
        if additional_args is None:
            return {}
        elif isinstance(additional_args, dict):
            return copy.copy(additional_args)
        elif isinstance(additional_args, str):
            try:
                return json.loads(additional_args)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in additional_args: {additional_args!r}. Error: {e}"
                )
        else:
            raise TypeError(
                f"additional_args must be None, dict, or str (JSON), got {type(additional_args).__name__}: {additional_args!r}"
            )

    def __enter__(self):
        """Context manager entry."""
        self.launch()
        self.wait_for_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.config.auto_shutdown:
            self.shutdown()
