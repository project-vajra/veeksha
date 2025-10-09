"""
Base server manager for orchestrating LLM inference servers.

This module provides the abstract base class for managing the lifecycle
of LLM inference servers (launch, health check, shutdown).
"""

import abc
import subprocess
import time
from typing import Optional

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
    
    @property
    def is_running(self) -> bool:
        """Check if server is currently running."""
        return self._is_running and self.process is not None and self.process.poll() is None
    
    @abc.abstractmethod
    def _build_launch_command(self) -> list[str]:
        """Build the command to launch the server.
        
        Returns:
            List of command arguments
        """
        pass
    
    def launch(self) -> bool:
        """Launch the inference server.
        
        Returns:
            True if launch was successful, False otherwise
        """
        if self.is_running:
            logger.warning(f"Server already running on {self.config.host}:{self.config.port}")
            return True
        
        try:
            command = self._build_launch_command()
            logger.info(f"Launching server with command: {' '.join(command)}")
            
            # Set up environment variables
            import os
            env = os.environ.copy()
            
            # Set CUDA_VISIBLE_DEVICES if gpu_ids specified
            gpu_env = self.config.get_gpu_env_var()
            if gpu_env is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_env
                logger.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_env}")
            
            # Launch server process
            self.process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
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
                # Try to get error output
                if self.process:
                    try:
                        stdout, stderr = self.process.communicate(timeout=5)
                        if stderr:
                            logger.error(f"Server stderr: {stderr}")
                        if stdout:
                            logger.info(f"Server stdout: {stdout}")
                    except Exception as e:
                        logger.error(f"Failed to read process output: {e}")
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
                    self.process.wait()
            
            self._is_running = False
            return True
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False
    
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
            import select
            
            stdout_lines = []
            stderr_lines = []
            
            # This is a simplified version - for production use proper logging
            return "", ""
            
        except Exception as e:
            logger.error(f"Error reading server logs: {e}")
            return "", ""
    
    def __enter__(self):
        """Context manager entry."""
        self.launch()
        self.wait_for_ready()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.config.auto_shutdown:
            self.shutdown()