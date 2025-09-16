"""Functional tests for veeksha benchmark functionality."""

import pytest

from .template_utils import create_benchmark_config
from .test_utils import BenchmarkTestRunner


@pytest.mark.functional
class TestBenchmarkFunctionality:
    """Test benchmark functionality with various configurations."""

    @pytest.mark.gpu
    def test_benchmark_with_poisson_interval_generator(
        self, temp_output_dir: str, sample_trace_file: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with Poisson request interval generator."""
        runner = BenchmarkTestRunner(temp_output_dir)
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

        runner.run_benchmark(config_content, "poisson_config.yml")

    @pytest.mark.gpu
    def test_benchmark_with_gamma_interval_generator(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with Gamma request interval generator."""
        runner = BenchmarkTestRunner(temp_output_dir)
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

        runner.run_benchmark(config_content, "gamma_config.yml")

    @pytest.mark.gpu
    def test_benchmark_with_static_interval_generator(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with static request interval generator."""
        runner = BenchmarkTestRunner(temp_output_dir)
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

        runner.run_benchmark(config_content, "static_config.yml")

    @pytest.mark.gpu
    def test_benchmark_with_zipf_length_generator(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with Zipf request length generator."""
        runner = BenchmarkTestRunner(temp_output_dir)
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

        runner.run_benchmark(config_content, "zipf_config.yml")

    @pytest.mark.gpu
    def test_benchmark_with_trace_length_generator(
        self, temp_output_dir: str, sample_trace_file: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with trace request length generator."""
        runner = BenchmarkTestRunner(temp_output_dir)
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

        runner.run_benchmark(config_content, "trace_config.yml")

    def test_benchmark_config_validation(self, temp_output_dir: str) -> None:
        """Test benchmark configuration validation."""
        runner = BenchmarkTestRunner(temp_output_dir)
        # Create invalid config with missing required fields
        config_content = """
timeout: 30
max_completed_requests: 1
# Missing client_config and request_generator_config
"""
        # This should fail due to missing config
        runner.run_benchmark(
            config_content,
            "invalid_config.yml",
            timeout=10,
            expected_return_code=1,
            check_output_files=False
        )