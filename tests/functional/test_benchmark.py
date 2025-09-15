"""Functional tests for veeksha benchmark functionality."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from .template_utils import create_benchmark_config


@pytest.mark.functional
class TestBenchmarkFunctionality:
    """Test benchmark functionality with various configurations."""

    @pytest.mark.gpu
    def test_benchmark_with_poisson_interval_generator(
        self, temp_output_dir: str, sample_trace_file: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with Poisson request interval generator."""
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=5,
            timeout=60,
            request_generator_type="synthetic",
            length_generator_type="fixed",
            interval_generator_type="poisson",
            prefill_tokens=30,
            decode_tokens=10,
            qps=0.5,
        )

        config_file = Path(temp_output_dir) / "poisson_config.yml"
        config_file.write_text(config_content)

        cmd = [
            "python", "-m", "veeksha.benchmark",
            "--benchmark-config-from-file", str(config_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

        # Check output files were created
        output_path = Path(temp_output_dir)
        assert output_path.exists()
        metrics_files = list(output_path.glob("*.json"))
        assert len(metrics_files) > 0, "No metrics files generated"

    @pytest.mark.gpu
    def test_benchmark_with_gamma_interval_generator(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with Gamma request interval generator."""
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=5,
            timeout=60,
            request_generator_type="synthetic",
            length_generator_type="uniform",
            interval_generator_type="gamma",
            min_tokens=20,
            max_tokens=60,
            qps=1.0,
            cv=0.5,
        )

        config_file = Path(temp_output_dir) / "gamma_config.yml"
        config_file.write_text(config_content)

        cmd = [
            "python", "-m", "veeksha.benchmark",
            "--benchmark-config-from-file", str(config_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

    @pytest.mark.gpu
    def test_benchmark_with_static_interval_generator(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with static request interval generator."""
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=30,
            request_generator_type="synthetic",
            length_generator_type="fixed",
            interval_generator_type="static",
            prefill_tokens=25,
            decode_tokens=15,
            duration=1.0,
        )

        config_file = Path(temp_output_dir) / "static_config.yml"
        config_file.write_text(config_content)

        cmd = [
            "python", "-m", "veeksha.benchmark",
            "--benchmark-config-from-file", str(config_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

    @pytest.mark.gpu
    def test_benchmark_with_zipf_length_generator(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with Zipf request length generator."""
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=30,
            request_generator_type="synthetic",
            length_generator_type="zipf",
            interval_generator_type="static",
            min_tokens=10,
            max_tokens=50,
            duration=1.0,
            theta=1.0,
            scramble=True,
        )

        config_file = Path(temp_output_dir) / "zipf_config.yml"
        config_file.write_text(config_content)

        cmd = [
            "python", "-m", "veeksha.benchmark",
            "--benchmark-config-from-file", str(config_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

    @pytest.mark.gpu
    def test_benchmark_with_trace_length_generator(
        self, temp_output_dir: str, sample_trace_file: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with trace request length generator."""
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=60,
            request_generator_type="synthetic",
            length_generator_type="trace",
            interval_generator_type="static",
            trace_file=sample_trace_file,
            max_tokens=512,
            duration=1.0,
        )

        config_file = Path(temp_output_dir) / "trace_config.yml"
        config_file.write_text(config_content)

        cmd = [
            "python", "-m", "veeksha.benchmark",
            "--benchmark-config-from-file", str(config_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

    def test_benchmark_config_validation(self, temp_output_dir: str) -> None:
        """Test benchmark configuration validation."""
        # Create invalid config with missing required fields
        config_content = """
timeout: 30
max_completed_requests: 1
# Missing client_config and request_generator_config
"""
        config_file = Path(temp_output_dir) / "invalid_config.yml"
        config_file.write_text(config_content)

        cmd = [
            "python", "-m", "veeksha.benchmark",
            "--benchmark-config-from-file", str(config_file),
        ]

        # This should fail due to missing config
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert result.returncode != 0, "Expected failure with invalid config"