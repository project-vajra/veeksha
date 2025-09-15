"""Functional tests for veeksha capacity search functionality."""

import subprocess
from pathlib import Path

import pytest

from .template_utils import create_capacity_search_config


@pytest.mark.functional
class TestCapacitySearch:
    """Test capacity search functionality with various SLO types."""

    @pytest.mark.gpu
    def test_capacity_search_tbt_ttft_slo(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test capacity search with TBT-TTFT based SLO."""
        slos = [
            {
                "type": "constant",
                "metric": "ttft",
                "value": 1.0,
                "percentile": 0.9,
                "name": "P90 TTFT",
            },
            {
                "type": "constant",
                "metric": "tbt",
                "value": 0.1,
                "percentile": 0.9,
                "name": "P90 TBT",
            },
        ]

        config_content = create_capacity_search_config(
            model=test_model,
            output_dir=f"{temp_output_dir}/capacity_search_results",
            slos=slos,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=30,
            max_iterations=2,
            prompt_length=30,
            output_length=15,
        )

        config_file = Path(temp_output_dir) / "capacity_search_config.yml"
        config_file.write_text(config_content)

        cmd = [
            "python",
            "-m",
            "veeksha.capacity_search",
            "--config-path",
            str(config_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Capacity search failed: {result.stderr}"

        # Check output files were created
        output_path = Path(temp_output_dir) / "capacity_search_results"
        assert output_path.exists(), "Output directory not created"

    @pytest.mark.gpu
    def test_capacity_search_ttft_tpot_slo(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test capacity search with TTFT-TPOT based SLO."""
        slos = [
            {
                "type": "constant",
                "metric": "ttft",
                "value": 1.0,
                "percentile": 0.9,
                "name": "P90 TTFT",
            },
            {
                "type": "constant",
                "metric": "tpot",
                "value": 0.1,
                "percentile": 0.9,
                "name": "P90 TPOT",
            },
        ]

        config_content = create_capacity_search_config(
            model=test_model,
            output_dir=f"{temp_output_dir}/capacity_search_results",
            slos=slos,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=30,
            max_iterations=2,
            prompt_length=30,
            output_length=15,
        )

        config_file = Path(temp_output_dir) / "capacity_search_config_tpot.yml"
        config_file.write_text(config_content)

        cmd = [
            "python",
            "-m",
            "veeksha.capacity_search",
            "--config-path",
            str(config_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Capacity search failed: {result.stderr}"

    @pytest.mark.gpu
    def test_capacity_search_with_custom_slos(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test capacity search with custom SLO configurations."""
        slos = [
            {
                "type": "constant",
                "metric": "ttft",
                "value": 0.5,
                "percentile": 0.5,
                "name": "P50 TTFT",
            },
            {
                "type": "constant",
                "metric": "tbt",
                "value": 0.05,
                "percentile": 0.95,
                "name": "P95 TBT",
            },
            {
                "type": "constant",
                "metric": "tpot",
                "value": 0.02,
                "percentile": 0.99,
                "name": "P99 TPOT",
            },
        ]

        config_content = create_capacity_search_config(
            model=test_model,
            output_dir=f"{temp_output_dir}/custom_slos_results",
            slos=slos,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=30,
            max_iterations=2,
            prompt_length=25,
            output_length=10,
        )

        config_file = Path(temp_output_dir) / "custom_slos_config.yml"
        config_file.write_text(config_content)

        cmd = [
            "python",
            "-m",
            "veeksha.capacity_search",
            "--config-path",
            str(config_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Capacity search failed: {result.stderr}"

    @pytest.mark.gpu
    def test_capacity_search_caching(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test that capacity search properly caches results between runs."""
        cache_dir = Path(temp_output_dir) / "cache_test_results"

        slos = [
            {
                "type": "constant",
                "metric": "ttft",
                "value": 1.0,
                "percentile": 0.9,
                "name": "P90 TTFT",
            }
        ]

        config_content = create_capacity_search_config(
            model=test_model,
            output_dir=str(cache_dir),
            slos=slos,
            api_url=vllm_server.base_url,
            max_completed_requests=2,
            timeout=20,
            max_iterations=1,
            prompt_length=20,
            output_length=10,
        )

        config_file = Path(temp_output_dir) / "cache_test_config.yml"
        config_file.write_text(config_content)

        # First run
        cmd = [
            "python",
            "-m",
            "veeksha.capacity_search",
            "--config-path",
            str(config_file),
        ]
        result1 = subprocess.run(cmd, capture_output=True, text=True)
        assert result1.returncode == 0, f"First capacity search failed: {result1.stderr}"

        # Check that cache directory was created
        assert cache_dir.exists(), "Cache directory not created"

        # Second run should use cached results
        result2 = subprocess.run(cmd, capture_output=True, text=True)
        assert result2.returncode == 0, f"Second capacity search failed: {result2.stderr}"

        # Check that cache files were created
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) > 0, "No cache files found"