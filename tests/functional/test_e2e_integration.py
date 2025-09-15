"""End-to-end integration tests with vLLM server."""

import subprocess
from pathlib import Path

import pytest
import requests

from veeksha.logger import init_logger

from .template_utils import create_benchmark_config, create_capacity_search_config

logger = init_logger("test_e2e_integration")

@pytest.mark.functional
class TestE2EIntegration:
    """Full E2E integration tests with vLLM server."""

    @pytest.mark.gpu
    def test_full_workflow_as_documented_in_readme(
        self, vllm_server, temp_output_dir: str, sample_trace_file: str
    ) -> None:
        """Test the complete workflow as documented in README."""

        # Test 1: Basic benchmark run
        logger.info("\n=== Running basic benchmark test ===")
        config_content = create_benchmark_config(
            model=vllm_server.model,
            output_dir=f"{temp_output_dir}/benchmark_basic",
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=60,
            request_generator_type="synthetic",
            length_generator_type="fixed",
            interval_generator_type="poisson",
            prefill_tokens=30,
            decode_tokens=15,
            qps=0.5,
            ttft_deadline=2.0,
            tbt_deadline=0.2,
        )

        config_file = Path(temp_output_dir) / "benchmark_config.yml"
        config_file.write_text(config_content)

        cmd = [
            "python", "-m", "veeksha.benchmark",
            "--benchmark-config-from-file", str(config_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

        # Verify output files
        output_path = Path(temp_output_dir) / "benchmark_basic"
        assert output_path.exists(), "Output directory not created"

        # Check for metrics files
        json_files = list(output_path.glob("*.json"))
        assert len(json_files) > 0, "No metrics files generated"

        # Test 2: Benchmark with trace file (if available)
        if Path(sample_trace_file).exists():
            logger.info("\n=== Running benchmark with trace file ===")
            trace_config = create_benchmark_config(
                model=vllm_server.model,
                output_dir=f"{temp_output_dir}/benchmark_trace",
                api_url=vllm_server.base_url,
                max_completed_requests=2,
                timeout=60,
                request_generator_type="synthetic",
                length_generator_type="trace",
                interval_generator_type="static",
                trace_file=sample_trace_file,
                max_tokens=512,
                duration=1.0,
            )

            trace_config_file = Path(temp_output_dir) / "trace_config.yml"
            trace_config_file.write_text(trace_config)

            cmd = [
                "python", "-m", "veeksha.benchmark",
                "--benchmark-config-from-file", str(trace_config_file),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"Trace benchmark failed: {result.stderr}"

        # Test 3: Capacity search
        logger.info("\n=== Running capacity search test ===")
        slos = [
            {
                "type": "constant",
                "metric": "ttft",
                "value": 2.0,
                "percentile": 0.9,
                "name": "P90 TTFT",
            }
        ]

        capacity_config = create_capacity_search_config(
            model=vllm_server.model,
            output_dir=f"{temp_output_dir}/capacity_search",
            slos=slos,
            api_url=vllm_server.base_url,
            max_completed_requests=2,
            timeout=60,
            max_iterations=2,
            prompt_length=25,
            output_length=10,
        )

        capacity_config_file = Path(temp_output_dir) / "capacity_config.yml"
        capacity_config_file.write_text(capacity_config)

        cmd = [
            "python", "-m", "veeksha.capacity_search",
            "--config-path", str(capacity_config_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Capacity search failed: {result.stderr}"

        # Verify capacity search output
        capacity_output = Path(temp_output_dir) / "capacity_search"
        assert capacity_output.exists(), "Capacity search output not created"

        logger.info("\n=== All E2E tests passed successfully ===")


@pytest.mark.functional
class TestVLLMServer:
    """Test with real vLLM server running."""

    @pytest.mark.gpu
    def test_vllm_server_starts_and_responds(self, vllm_server) -> None:
        """Test that vLLM server starts and responds to requests."""
        # Test models endpoint
        response = requests.get(f"{vllm_server.base_url}/models")
        assert response.status_code == 200
        models = response.json()
        assert "data" in models
        assert len(models["data"]) > 0

        # Test chat completions endpoint
        chat_request = {
            "model": vllm_server.model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "temperature": 0.0,
        }

        response = requests.post(
            f"{vllm_server.base_url}/chat/completions",
            json=chat_request,
        )

        if response.status_code != 200:
            logger.info(f"❌ Chat completions failed with status {response.status_code}")
            logger.info(f"Response: {response.text}")
        assert response.status_code == 200
        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0
        assert "message" in result["choices"][0]

    @pytest.mark.gpu
    def test_benchmark_with_vllm(self, vllm_server, temp_output_dir: str) -> None:
        """Test benchmark with vLLM server."""

        # Run a simple benchmark using config
        config_content = create_benchmark_config(
            model=vllm_server.model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=2,
            timeout=30,
            request_generator_type="synthetic",
            length_generator_type="fixed",
            interval_generator_type="static",
            prefill_tokens=20,
            decode_tokens=10,
            duration=2.0,
        )

        config_file = Path(temp_output_dir) / "simple_benchmark_config.yml"
        config_file.write_text(config_content)

        cmd = [
            "python", "-m", "veeksha.benchmark",
            "--benchmark-config-from-file", str(config_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

        # Check metrics were generated
        output_path = Path(temp_output_dir)
        json_files = list(output_path.glob("*.json"))
        assert len(json_files) > 0, "No metrics files generated"
