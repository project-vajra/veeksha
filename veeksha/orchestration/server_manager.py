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
from dataclasses import replace
from pathlib import Path
from typing import IO, Any, Dict, Optional

import requests

from veeksha.config.server import BaseServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration.resource_manager import ResourceManager

logger = init_logger(__name__)


class BaseServerManager(abc.ABC):
    """Abstract base class for managing LLM inference servers.

    Subclasses should implement engine-specific launch commands and
    health check logic.
    """

    def __init__(self, config: BaseServerConfig, output_dir: Optional[str] = None):
        """Initialize the server manager.

        Args:
            config: Server configuration
            output_dir: Directory for server logs.
        """
        self.config: BaseServerConfig = config
        self.output_dir = output_dir
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

        output_dir = self.output_dir or os.environ.get("VEEKSHA_OUTPUT_DIR")
        if output_dir:
            log_dir = Path(output_dir)
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / log_filename
                log_file = open(log_path, "w+", encoding="utf-8")
                self._log_file_path = log_path
                self._delete_log_file_on_cleanup = False
                logger.info(f"Server logs will be written to: {log_path}")
                return log_file
            except Exception as exc:  # pragma: no cover - fallback path
                logger.warning(
                    "Unable to create server log file in output directory "
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
            # auto-allocate if not specified
            if self.config.gpu_ids is None:
                num_gpus = self.config.get_num_gpus()

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

                self._allocated_job_id = job_id
                gpu_ids = [gpu_id for _, gpu_id in resource_mapping]
                self.config = replace(self.config, gpu_ids=gpu_ids)

            command = self._build_launch_command()
            logger.info(f"Launching server with command: {' '.join(command)}")

            env = os.environ.copy()

            env_path = getattr(self.config, "env_path", None)
            if env_path:
                # platform-specific scripts directory
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

            gpu_env = self.config.get_gpu_env_var()
            if gpu_env is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_env

            self._log_file = self._create_log_file()

            # launch server
            self.process = subprocess.Popen(
                command,
                env=env,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

            logger.info(f"Server process started with PID: {self.process.pid}")
            self._is_running = True
            return True, None

        except Exception as e:
            # release GPUs
            if self._allocated_job_id is not None:
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
            if not self.is_running:
                logger.error("Server process terminated unexpectedly")
                # check common errors
                if self._log_file:
                    try:
                        self._log_file.seek(0)
                        logs = self._log_file.read()

                        # GPU memory error
                        if (
                            "Free memory on device" in logs
                            and "is less than desired GPU memory utilization" in logs
                        ):
                            import re

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
                                    "GPU memory detected but couldn't parse details"
                                )
                        else:
                            log_lines = logs.strip().split("\n")
                            recent_logs = "\n".join(log_lines[-50:])
                            logger.error(f"Recent server logs:\n{recent_logs}")
                    except Exception as e:
                        logger.error(f"Failed to read server logs: {e}")
                return False

            if self.health_check():
                return True

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
            if self.process is None:
                logger.error("Server process is None, cannot shutdown")
                success = False
            else:
                if force:
                    self.process.kill()
                # graceful shutdown
                else:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            "Server did not shut down gracefully, force killing"
                        )
                        self.process.kill()

                try:
                    self.process.wait(timeout=5)
                except Exception as e:
                    logger.warning(f"Error waiting for process to exit: {e}")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            success = False
        finally:
            # reset state and clean up resources, even with exceptions
            self._is_running = False

            if self._allocated_job_id is not None:
                try:
                    self.resource_manager.release_resources(self._allocated_job_id)
                except Exception as e:
                    logger.error(f"Error releasing resources: {e}")
                finally:
                    self._allocated_job_id = None

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
        log_path: Optional[Path] = None
        if self._log_file_path is not None:
            log_path = self._log_file_path
        elif self._log_file is not None:
            log_path = Path(self._log_file.name)
        else:
            return "", ""

        try:
            if self._log_file is not None:
                try:
                    self._log_file.flush()
                except Exception:
                    pass

            if not log_path.exists():
                return "", ""

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
        self.shutdown()
