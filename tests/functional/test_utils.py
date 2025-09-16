"""Common utilities for functional tests to reduce code duplication."""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import tempfile


class BenchmarkTestRunner:
    """Helper class to run benchmark tests with less boilerplate."""

    def __init__(self, temp_output_dir: str):
        self.temp_output_dir = Path(temp_output_dir)

    def run_benchmark(
        self,
        config_content: str,
        config_name: str = "config.yml",
        timeout: Optional[int] = None,
        check_output_files: bool = True,
        expected_return_code: int = 0,
    ) -> subprocess.CompletedProcess:
        """Run a benchmark test with the given configuration.

        Args:
            config_content: YAML configuration content
            config_name: Name for the config file
            timeout: Optional timeout for subprocess
            check_output_files: Whether to verify output files were created
            expected_return_code: Expected return code (0 for success)

        Returns:
            The completed process result
        """
        config_file = self.temp_output_dir / config_name
        config_file.write_text(config_content)

        cmd = [
            "python", "-m", "veeksha.benchmark",
            "--benchmark-config-from-file", str(config_file),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            # For invalid configs, timeout is expected
            if expected_return_code != 0:
                return subprocess.CompletedProcess(cmd, 1, '', 'Command timed out')
            raise

        assert result.returncode == expected_return_code, (
            f"Benchmark failed with code {result.returncode}: {result.stderr}"
        )

        if check_output_files and expected_return_code == 0:
            self._verify_output_files()

        return result

    def _verify_output_files(self) -> None:
        """Verify that output files were created."""
        assert self.temp_output_dir.exists(), "Output directory not created"
        metrics_files = list(self.temp_output_dir.glob("**/*.json"))
        assert len(metrics_files) > 0, "No metrics files generated"


class CapacitySearchTestRunner:
    """Helper class to run capacity search tests with less boilerplate."""

    def __init__(self, temp_output_dir: str):
        self.temp_output_dir = Path(temp_output_dir)

    def run_capacity_search(
        self,
        config_content: str,
        config_name: str = "capacity_config.yml",
        output_subdir: str = "capacity_search_results",
        timeout: Optional[int] = None,
        check_output: bool = True,
        expected_return_code: int = 0,
    ) -> subprocess.CompletedProcess:
        """Run a capacity search test with the given configuration.

        Args:
            config_content: YAML configuration content
            config_name: Name for the config file
            output_subdir: Subdirectory for output
            timeout: Optional timeout for subprocess
            check_output: Whether to verify output directory was created
            expected_return_code: Expected return code (0 for success)

        Returns:
            The completed process result
        """
        config_file = self.temp_output_dir / config_name
        config_file.write_text(config_content)

        cmd = [
            "python", "-m", "veeksha.capacity_search",
            "--capacity-search-config-from-file", str(config_file),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        assert result.returncode == expected_return_code, (
            f"Capacity search failed with code {result.returncode}: {result.stderr}"
        )

        if check_output and expected_return_code == 0:
            output_path = self.temp_output_dir / output_subdir
            assert output_path.exists(), f"Output directory {output_path} not created"

        return result


def create_slo_config(
    metric: str,
    value: float,
    percentile: float = 0.9,
    slo_type: str = "constant",
    name: Optional[str] = None
) -> Dict[str, Any]:
    """Create a single SLO configuration dictionary.

    Args:
        metric: The metric type (ttft, tbt, tpot)
        value: The threshold value
        percentile: The percentile (default 0.9)
        slo_type: The SLO type (default "constant")
        name: Optional descriptive name

    Returns:
        SLO configuration dictionary
    """
    slo = {
        "type": slo_type,
        "metric": metric,
        "value": value,
        "percentile": percentile,
    }

    if name is None:
        name = f"P{int(percentile * 100)} {metric.upper()}"
    slo["name"] = name

    return slo


def create_standard_slos(
    ttft_value: float = 1.0,
    tbt_value: Optional[float] = None,
    tpot_value: Optional[float] = None,
    percentile: float = 0.9
) -> List[Dict[str, Any]]:
    """Create a standard set of SLOs for testing.

    Args:
        ttft_value: TTFT threshold value
        tbt_value: Optional TBT threshold value
        tpot_value: Optional TPOT threshold value
        percentile: Percentile for all SLOs

    Returns:
        List of SLO configurations
    """
    slos = [create_slo_config("ttft", ttft_value, percentile)]

    if tbt_value is not None:
        slos.append(create_slo_config("tbt", tbt_value, percentile))

    if tpot_value is not None:
        slos.append(create_slo_config("tpot", tpot_value, percentile))

    return slos


def verify_cache_behavior(
    runner: CapacitySearchTestRunner,
    config_content: str,
    cache_dir: Path
) -> None:
    """Verify that capacity search properly caches results between runs.

    Args:
        runner: The capacity search test runner
        config_content: Configuration content to use
        cache_dir: Directory where cache should be created
    """
    # First run
    result1 = runner.run_capacity_search(
        config_content,
        config_name="cache_test_config.yml",
        output_subdir=cache_dir.name,
        check_output=False
    )

    # Verify cache was created
    assert cache_dir.exists(), "Cache directory not created"
    cache_files_before = list(cache_dir.glob("**/*.json"))
    assert len(cache_files_before) > 0, "No cache files found after first run"

    # Second run should use cached results
    result2 = runner.run_capacity_search(
        config_content,
        config_name="cache_test_config.yml",
        output_subdir=cache_dir.name,
        check_output=False
    )

    # Verify cache files still exist
    cache_files_after = list(cache_dir.glob("**/*.json"))
    assert len(cache_files_after) > 0, "Cache files missing after second run"