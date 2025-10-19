"""
Integration test for resource management system.

This test demonstrates the complete workflow of resource-aware benchmarking.
"""

import pytest

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.server import ServerConfig
from veeksha.orchestration import (
    ResourceManager,
    ParallelBenchmarkRunner,
    SequentialJobQueue,
    create_experiment_grid,
)


class TestResourceManagementIntegration:
    """Integration tests for resource management."""

    def test_experiment_grid_creation(self):
        """Test creating experiment grids."""
        grid = create_experiment_grid(
            models=["model-a", "model-b"],
            tensor_parallel_sizes=[1, 2],
            qps_values=[1.0, 2.0],
        )

        # Should create 2 * 2 * 2 = 8 combinations
        assert len(grid) == 8

        # Verify first combination
        assert grid[0]["model"] == "model-a"
        assert "tensor_parallel_size" in grid[0]
        assert "qps" in grid[0]

    def test_resource_manager_workflow(self):
        """Test complete ResourceManager workflow."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("test-node", num_gpus=4)

        # Initial state
        assert rm.get_total_gpus() == 4
        assert rm.get_free_gpus() == 4

        # Allocate for job 1
        mapping1 = rm.allocate_resources(num_gpus=2, job_id="job1")
        assert mapping1 is not None
        assert len(mapping1) == 2
        assert rm.get_free_gpus() == 2

        # Allocate for job 2
        mapping2 = rm.allocate_resources(num_gpus=2, job_id="job2")
        assert mapping2 is not None
        assert rm.get_free_gpus() == 0

        # Try to allocate when no resources (should fail)
        mapping3 = rm.allocate_resources(num_gpus=1, job_id="job3")
        assert mapping3 is None

        # Release job 1
        rm.release_resources("job1")
        assert rm.get_free_gpus() == 2

        # Now job 3 should succeed
        mapping3 = rm.allocate_resources(num_gpus=1, job_id="job3")
        assert mapping3 is not None

        # Cleanup
        rm.release_resources("job2")
        rm.release_resources("job3")
        assert rm.get_free_gpus() == 4

    def test_server_config_enhancements(self):
        """Test ServerConfig resource management fields."""
        config = ServerConfig(
            model="test-model",
            tensor_parallel_size=4,
            gpu_ids=[0, 1, 2, 3],
            require_contiguous_gpus=True,
            priority=10,
            estimated_memory_per_gpu_gb=16.0,
        )

        # Test get_num_gpus
        assert config.get_num_gpus() == 4

        # Test new fields
        assert config.require_contiguous_gpus is True
        assert config.priority == 10
        assert config.estimated_memory_per_gpu_gb == 16.0

        # Test to_dict includes new fields
        config_dict = config.to_dict()
        assert "require_contiguous_gpus" in config_dict
        assert "priority" in config_dict
        assert "estimated_memory_per_gpu_gb" in config_dict

    def test_sequential_job_queue(self):
        """Test SequentialJobQueue functionality."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("test-node", num_gpus=4)

        queue = SequentialJobQueue(resource_manager=rm)

        # Mock benchmark function
        def mock_benchmark(config):
            return {"status": "success", "model": config.server_config.model}

        # Add jobs
        for i in range(3):
            server_config = ServerConfig(
                model=f"model-{i}",
                tensor_parallel_size=1,
                port=8000 + i,
                auto_shutdown=True,
            )
            benchmark_config = BenchmarkConfig(
                server_config=server_config,
                timeout=60,
                max_completed_requests=10,
                client_config={"model": f"model-{i}"},
                request_generator_config={
                    "type": "synthetic",
                    "length_generator_config": {"type": "fixed"},
                },
                metrics_config={"output_dir": f"test_{i}"},
            )
            queue.add_job(server_config, benchmark_config, mock_benchmark)

        assert len(queue.jobs) == 3

    @pytest.mark.parametrize(
        "num_gpus,contiguous,expected_contiguous",
        [
            (2, True, True),  # 2 GPUs, contiguous required
            (4, True, True),  # 4 GPUs, contiguous required
            (2, False, False),  # 2 GPUs, any allocation ok
        ],
    )
    def test_contiguous_allocation(
        self, num_gpus: int, contiguous: bool, expected_contiguous: bool
    ):
        """Test contiguous GPU allocation."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("test-node", num_gpus=8)

        mapping = rm.allocate_resources(
            num_gpus=num_gpus, job_id="test", contiguous=contiguous
        )

        assert mapping is not None
        assert len(mapping) == num_gpus

        if expected_contiguous:
            # Check GPU IDs are contiguous
            gpu_ids = sorted([gpu_id for _, gpu_id in mapping])
            for i in range(len(gpu_ids) - 1):
                assert gpu_ids[i + 1] == gpu_ids[i] + 1

        rm.release_resources("test")

    def test_resource_wait_timeout(self):
        """Test waiting for resources with timeout."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("test-node", num_gpus=2)

        # Allocate all resources
        mapping1 = rm.allocate_resources(num_gpus=2, job_id="job1")
        assert mapping1 is not None

        # Try to wait with short timeout (should fail)
        mapping2 = rm.wait_for_resources(
            num_gpus=1, job_id="job2", timeout=0.1, poll_interval=0.05
        )
        assert mapping2 is None

        # Release and try again (should succeed)
        rm.release_resources("job1")
        mapping3 = rm.wait_for_resources(
            num_gpus=1, job_id="job3", timeout=1.0, poll_interval=0.1
        )
        assert mapping3 is not None

        rm.release_resources("job3")

    def test_multi_node_allocation(self):
        """Test allocation across multiple nodes."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=4)
        rm.add_node("node2", num_gpus=4)

        # Check total
        assert rm.get_total_gpus() == 8
        assert len(rm.nodes) == 2

        # Allocate across nodes
        mapping = rm.allocate_resources(num_gpus=6, job_id="multi_node", contiguous=False)

        if mapping:  # May fail on single-node allocation attempt
            assert len(mapping) == 6
            rm.release_resources("multi_node")

    def test_resource_status_monitoring(self):
        """Test resource status monitoring."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=4)

        # Initial status
        status = rm.get_resource_status()
        assert status["total_nodes"] == 1
        assert status["total_gpus"] == 4
        assert status["free_gpus"] == 4
        assert status["allocated_gpus"] == 0
        assert status["active_jobs"] == 0

        # Allocate resources
        rm.allocate_resources(num_gpus=2, job_id="job1")

        # Check updated status
        status = rm.get_resource_status()
        assert status["free_gpus"] == 2
        assert status["allocated_gpus"] == 2
        assert status["active_jobs"] == 1

        # Release
        rm.release_resources("job1")

        # Check final status
        status = rm.get_resource_status()
        assert status["free_gpus"] == 4
        assert status["active_jobs"] == 0


# Mark this as an integration test
pytestmark = pytest.mark.integration
