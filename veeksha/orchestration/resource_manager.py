"""
Resource manager for GPU allocation and management.

This module provides resource-aware scheduling of LLM inference servers,
enabling efficient utilization of GPU resources across multiple experiments.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from veeksha.logger import init_logger

logger = init_logger(__name__)


# Type aliases for clarity
ResourceMapping = List[Tuple[str, int]]  # List of (node_hostname, gpu_id)


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    node_hostname: str
    gpu_id: int
    total_memory_mb: int
    is_free: bool = True


@dataclass
class NodeInfo:
    """Information about a compute node."""

    hostname: str
    num_gpus: int
    gpus: List[GPUInfo] = field(default_factory=list)
    is_fully_free: bool = True


class ResourceManager:
    """Manager for tracking and allocating GPU resources.

    This class provides resource-aware scheduling for LLM inference servers,
    enabling efficient utilization of GPUs across multiple experiments.

    Features:
    - Automatic GPU detection
    - Contiguous GPU allocation on single nodes
    - Multi-node allocation for large jobs
    - Resource tracking and cleanup
    """

    def __init__(self, detect_gpus: bool = True):
        """Initialize the resource manager.

        Args:
            detect_gpus: If True, automatically detect available GPUs using nvidia-smi
        """
        self.nodes: Dict[str, NodeInfo] = {}
        self.allocated_resources: Dict[str, ResourceMapping] = {}  # job_id -> resources

        if detect_gpus:
            self._detect_gpus()

    def _detect_gpus(self) -> None:
        """Detect available GPUs using Ray and check their memory availability."""
        try:
            import ray

            ray.init(ignore_reinit_error=True)
            nodes = ray.nodes()
            for node in nodes:
                node_ip = node["NodeManagerAddress"]
                num_gpus = int(node["Resources"].get("GPU", 0))
                if num_gpus > 0:
                    # Get GPU memory info using nvidia-smi
                    gpu_memory_info = self._get_gpu_memory_info()
                    
                    gpus = []
                    for i in range(num_gpus):
                        total_memory_mb = 0
                        is_free = True
                        
                        if i in gpu_memory_info:
                            total_memory_mb = int(gpu_memory_info[i]["total"])
                            free_memory_mb = gpu_memory_info[i]["free"]
                            # Mark as free only if >= 90% of memory is available
                            is_free = (free_memory_mb / total_memory_mb) >= 0.90
                            if not is_free:
                                logger.warning(
                                    f"GPU {i} on node {node_ip} has only "
                                    f"{free_memory_mb / total_memory_mb * 100:.1f}% free memory "
                                    f"({free_memory_mb:.0f}/{total_memory_mb:.0f} MB), marking as unavailable"
                                )
                        
                        gpus.append(
                            GPUInfo(
                                node_hostname=node_ip,
                                gpu_id=i,
                                total_memory_mb=total_memory_mb,
                                is_free=is_free,
                            )
                        )
                    
                    self.nodes[node_ip] = NodeInfo(
                        hostname=node_ip,
                        num_gpus=num_gpus,
                        gpus=gpus,
                        is_fully_free=all(gpu.is_free for gpu in gpus),
                    )
                    free_gpus = [g for g in gpus if g.is_free]
                    logger.info(
                        f"Detected {num_gpus} GPUs on node {node_ip}, "
                        f"{len(free_gpus)} available (>=90% free): "
                        f"{[f'GPU{g.gpu_id}' for g in free_gpus]}"
                    )
        except ImportError:
            logger.error("Ray not installed. Cannot detect GPUs.")
        except Exception as e:
            logger.error(f"Error detecting GPUs with Ray: {e}")

    def _get_gpu_memory_info(self) -> Dict[int, Dict[str, float]]:
        """Get GPU memory information using nvidia-smi.
        
        Returns:
            Dictionary mapping GPU ID to memory info (total, free, used in MB)
        """
        try:
            import subprocess
            
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.total,memory.free,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            
            gpu_info = {}
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    gpu_id = int(parts[0])
                    total_mb = float(parts[1])
                    free_mb = float(parts[2])
                    used_mb = float(parts[3])
                    gpu_info[gpu_id] = {
                        "total": total_mb,
                        "free": free_mb,
                        "used": used_mb,
                    }
            
            return gpu_info
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {}

    def add_node(
        self, hostname: str, num_gpus: int, gpu_memory_mb: Optional[int] = None
    ) -> None:
        """Manually add a node with GPUs.

        Args:
            hostname: Hostname of the node
            num_gpus: Number of GPUs on the node
            gpu_memory_mb: Memory per GPU in MB (optional)
        """
        gpus = []
        for gpu_id in range(num_gpus):
            gpu_info = GPUInfo(
                node_hostname=hostname,
                gpu_id=gpu_id,
                total_memory_mb=gpu_memory_mb or 0,
                is_free=True,
            )
            gpus.append(gpu_info)

        self.nodes[hostname] = NodeInfo(
            hostname=hostname, num_gpus=num_gpus, gpus=gpus, is_fully_free=True
        )
        logger.info(f"Added node {hostname} with {num_gpus} GPUs")

    def get_total_gpus(self) -> int:
        """Get total number of GPUs across all nodes."""
        return sum(node.num_gpus for node in self.nodes.values())

    def get_free_gpus(self) -> int:
        """Get number of free GPUs across all nodes."""
        return sum(
            sum(1 for gpu in node.gpus if gpu.is_free) for node in self.nodes.values()
        )

    def allocate_resources(
        self, num_gpus: int, job_id: Optional[str] = None, contiguous: bool = True
    ) -> Optional[ResourceMapping]:
        """Allocate GPUs for a job.

        Args:
            num_gpus: Number of GPUs to allocate
            job_id: Unique identifier for the job (auto-generated if None)
            contiguous: If True, allocate contiguous GPUs on same node

        Returns:
            ResourceMapping of allocated (hostname, gpu_id) pairs, or None if allocation failed
        """
        if num_gpus <= 0:
            logger.error(f"Invalid num_gpus: {num_gpus}")
            return None

        if num_gpus > self.get_total_gpus():
            logger.error(
                f"Requested {num_gpus} GPUs, but only {self.get_total_gpus()} available in cluster"
            )
            return None

        if num_gpus > self.get_free_gpus():
            logger.warning(
                f"Requested {num_gpus} GPUs, but only {self.get_free_gpus()} currently free"
            )
            return None

        # Try to allocate on a single node first
        for node in self.nodes.values():
            free_gpus = [gpu for gpu in node.gpus if gpu.is_free]

            if len(free_gpus) >= num_gpus:
                # Check for contiguous allocation if required
                if contiguous:
                    allocated = self._allocate_contiguous(free_gpus, num_gpus)
                else:
                    allocated = free_gpus[:num_gpus]

                if allocated:
                    resource_mapping = [
                        (gpu.node_hostname, gpu.gpu_id) for gpu in allocated
                    ]

                    # Mark GPUs as allocated
                    for gpu in allocated:
                        gpu.is_free = False

                    # Update node status
                    node.is_fully_free = all(gpu.is_free for gpu in node.gpus)

                    # Track allocation
                    if job_id is None:
                        job_id = f"job_{int(time.time() * 1000)}"
                    self.allocated_resources[job_id] = resource_mapping

                    logger.info(
                        f"Allocated {num_gpus} GPUs for {job_id}: "
                        f"{[(h, g) for h, g in resource_mapping]}"
                    )
                    return resource_mapping

        # Multi-node allocation (if single-node failed and we have multiple nodes)
        if len(self.nodes) > 1 and not contiguous:
            allocated_gpus = []
            for node in self.nodes.values():
                if len(allocated_gpus) >= num_gpus:
                    break

                free_gpus = [gpu for gpu in node.gpus if gpu.is_free]
                remaining_needed = num_gpus - len(allocated_gpus)
                allocated_gpus.extend(free_gpus[:remaining_needed])

            if len(allocated_gpus) == num_gpus:
                resource_mapping = [
                    (gpu.node_hostname, gpu.gpu_id) for gpu in allocated_gpus
                ]

                # Mark GPUs as allocated
                for gpu in allocated_gpus:
                    gpu.is_free = False

                # Update all node statuses
                for node in self.nodes.values():
                    node.is_fully_free = all(gpu.is_free for gpu in node.gpus)

                # Track allocation
                if job_id is None:
                    job_id = f"job_{int(time.time() * 1000)}"
                self.allocated_resources[job_id] = resource_mapping

                logger.info(
                    f"Allocated {num_gpus} GPUs across multiple nodes for {job_id}"
                )
                return resource_mapping

        logger.warning(f"Could not allocate {num_gpus} GPUs")
        return None

    def get_gpu_memory_mb(self, resource_mapping: ResourceMapping) -> int:
        """Get total GPU memory for allocated resources.

        Args:
            resource_mapping: List of (hostname, gpu_id) tuples

        Returns:
            Total GPU memory in MB across all allocated GPUs
        """
        total_memory = 0
        for hostname, gpu_id in resource_mapping:
            if hostname in self.nodes:
                node = self.nodes[hostname]
                for gpu in node.gpus:
                    if gpu.gpu_id == gpu_id:
                        total_memory += gpu.total_memory_mb
                        break
        return total_memory

    def _allocate_contiguous(
        self, free_gpus: List[GPUInfo], num_gpus: int
    ) -> Optional[List[GPUInfo]]:
        """Try to allocate contiguous GPUs.

        Args:
            free_gpus: List of free GPUs to choose from
            num_gpus: Number of GPUs to allocate

        Returns:
            List of allocated GPUs if successful, None otherwise
        """
        # Sort by GPU ID
        sorted_gpus = sorted(free_gpus, key=lambda g: g.gpu_id)

        # Find contiguous blocks
        for i in range(len(sorted_gpus) - num_gpus + 1):
            # Check if this starting position gives us contiguous GPUs
            candidate = sorted_gpus[i : i + num_gpus]
            gpu_ids = [g.gpu_id for g in candidate]

            # Check if GPU IDs are contiguous
            if gpu_ids == list(range(gpu_ids[0], gpu_ids[0] + num_gpus)):
                return candidate

        # If contiguous allocation failed, return any num_gpus
        logger.debug(
            "Could not find contiguous GPUs, falling back to non-contiguous allocation"
        )
        return sorted_gpus[:num_gpus]

    def release_resources(self, job_id: str) -> bool:
        """Release resources allocated to a job.

        Args:
            job_id: Job identifier

        Returns:
            True if resources were released, False if job_id not found
        """
        if job_id not in self.allocated_resources:
            logger.warning(f"No allocation found for job_id: {job_id}")
            return False

        resource_mapping = self.allocated_resources[job_id]

        # Free the GPUs
        for hostname, gpu_id in resource_mapping:
            if hostname in self.nodes:
                node = self.nodes[hostname]
                for gpu in node.gpus:
                    if gpu.gpu_id == gpu_id:
                        gpu.is_free = True
                        break

                # Update node status
                node.is_fully_free = all(gpu.is_free for gpu in node.gpus)

        # Remove from tracking
        del self.allocated_resources[job_id]

        logger.info(f"Released resources for {job_id}")
        return True

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status.

        Returns:
            Dictionary with resource information
        """
        status = {
            "total_nodes": len(self.nodes),
            "total_gpus": self.get_total_gpus(),
            "free_gpus": self.get_free_gpus(),
            "allocated_gpus": self.get_total_gpus() - self.get_free_gpus(),
            "active_jobs": len(self.allocated_resources),
            "nodes": {},
        }

        for hostname, node in self.nodes.items():
            node_status = {
                "num_gpus": node.num_gpus,
                "free_gpus": sum(1 for gpu in node.gpus if gpu.is_free),
                "fully_free": node.is_fully_free,
                "gpus": [
                    {
                        "gpu_id": gpu.gpu_id,
                        "free": gpu.is_free,
                        "memory_mb": gpu.total_memory_mb,
                    }
                    for gpu in node.gpus
                ],
            }
            status["nodes"][hostname] = node_status

        return status

    def wait_for_resources(
        self,
        num_gpus: int,
        timeout: Optional[float] = None,
        poll_interval: float = 1.0,
        job_id: Optional[str] = None,
    ) -> Optional[ResourceMapping]:
        """Wait for resources to become available.

        Args:
            num_gpus: Number of GPUs needed
            timeout: Maximum time to wait in seconds (None = wait indefinitely)
            poll_interval: Time between checks in seconds
            job_id: Job identifier for allocation

        Returns:
            ResourceMapping if successful, None if timeout
        """
        start_time = time.time()

        while True:
            # Try to allocate
            resource_mapping = self.allocate_resources(num_gpus, job_id=job_id)
            if resource_mapping:
                return resource_mapping

            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                logger.warning(f"Timeout waiting for {num_gpus} GPUs after {timeout}s")
                return None

            # Wait before next attempt
            logger.debug(
                f"Waiting for {num_gpus} GPUs... (free: {self.get_free_gpus()})"
            )
            time.sleep(poll_interval)
