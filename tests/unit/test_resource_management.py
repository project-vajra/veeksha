"""
Tests for resource management functionality.
"""

import pytest

from veeksha.orchestration.resource_manager import ResourceManager


class TestResourceManager:
    """Test ResourceManager basic functionality."""

    def test_manual_node_addition(self):
        """Test manually adding nodes and allocating resources."""
        rm = ResourceManager(detect_gpus=False)  # Don't auto-detect

        # Add a node manually
        rm.add_node("node1", num_gpus=4)

        # Check status
        status = rm.get_resource_status()
        assert status["total_gpus"] == 4
        assert status["free_gpus"] == 4
        assert len(status["nodes"]) == 1

    def test_resource_allocation(self):
        """Test allocating and releasing resources."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=4)

        # Allocate 2 GPUs
        mapping = rm.allocate_resources(num_gpus=2, job_id="job1")
        assert mapping is not None
        assert len(mapping) == 2

        # Check free GPUs reduced
        assert rm.get_free_gpus() == 2

        # Release resources
        success = rm.release_resources("job1")
        assert success is True

        # Check GPUs are free again
        assert rm.get_free_gpus() == 4

    def test_contiguous_allocation(self):
        """Test contiguous GPU allocation."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=8)

        # Allocate 4 contiguous GPUs
        mapping = rm.allocate_resources(num_gpus=4, job_id="job1", contiguous=True)
        assert mapping is not None
        assert len(mapping) == 4

        # Check GPU IDs are contiguous
        gpu_ids = sorted([gpu_id for _, gpu_id in mapping])
        assert gpu_ids == [0, 1, 2, 3] or gpu_ids == [4, 5, 6, 7]

        rm.release_resources("job1")

    def test_allocation_failure(self):
        """Test allocation when insufficient resources."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=2)

        # Try to allocate more GPUs than available
        mapping = rm.allocate_resources(num_gpus=4, job_id="job1")
        assert mapping is None

    def test_multiple_allocations(self):
        """Test multiple concurrent allocations."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=4)

        # Allocate for multiple jobs
        mapping1 = rm.allocate_resources(num_gpus=2, job_id="job1")
        mapping2 = rm.allocate_resources(num_gpus=2, job_id="job2")

        assert mapping1 is not None
        assert mapping2 is not None
        assert rm.get_free_gpus() == 0

        # Release one job
        rm.release_resources("job1")
        assert rm.get_free_gpus() == 2

        # Release other job
        rm.release_resources("job2")
        assert rm.get_free_gpus() == 4

    def test_wait_for_resources(self):
        """Test waiting for resources to become available."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=2)

        # Allocate all GPUs
        mapping1 = rm.allocate_resources(num_gpus=2, job_id="job1")
        assert mapping1 is not None

        # Try to wait with very short timeout (should fail)
        mapping2 = rm.wait_for_resources(
            num_gpus=1, job_id="job2", timeout=0.1, poll_interval=0.05
        )
        assert mapping2 is None

        # Release resources
        rm.release_resources("job1")

        # Now allocation should succeed
        mapping3 = rm.allocate_resources(num_gpus=1, job_id="job3")
        assert mapping3 is not None

        rm.release_resources("job3")

    @pytest.mark.skipif(
        not pytest.config.getoption("--run-gpu-tests", default=False),
        reason="GPU detection tests require --run-gpu-tests flag",
    )
    def test_gpu_detection(self):
        """Test automatic GPU detection (requires GPUs)."""
        rm = ResourceManager(detect_gpus=True)

        status = rm.get_resource_status()
        # Should detect at least some GPUs if running on GPU machine
        assert status["total_gpus"] >= 0

        if status["total_gpus"] > 0:
            # Test allocation with detected GPUs
            mapping = rm.allocate_resources(num_gpus=1, job_id="test")
            assert mapping is not None
            rm.release_resources("test")


class TestServerConfigEnhancements:
    """Test ServerConfig resource management fields."""

    def test_server_config_num_gpus(self):
        """Test get_num_gpus method."""
        from veeksha.config.server import ServerConfig

        # Test with tensor_parallel_size
        config1 = ServerConfig(tensor_parallel_size=4)
        assert config1.get_num_gpus() == 4

        # Test with explicit gpu_ids
        config2 = ServerConfig(gpu_ids=[0, 1, 2])
        assert config2.get_num_gpus() == 3

    def test_server_config_resource_fields(self):
        """Test new resource management fields."""
        from veeksha.config.server import ServerConfig

        config = ServerConfig(
            model="test-model",
            tensor_parallel_size=2,
            require_contiguous_gpus=True,
            priority=10,
            estimated_memory_per_gpu_gb=16.0,
        )

        assert config.require_contiguous_gpus is True
        assert config.priority == 10
        assert config.estimated_memory_per_gpu_gb == 16.0

        # Test to_dict includes new fields
        config_dict = config.to_dict()
        assert "require_contiguous_gpus" in config_dict
        assert "priority" in config_dict
        assert "estimated_memory_per_gpu_gb" in config_dict
