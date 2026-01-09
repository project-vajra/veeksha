"""Unit tests for CLI commands."""

import subprocess

import pytest


@pytest.mark.unit
class TestCLI:
    """Test CLI commands work correctly."""

    def test_benchmark_help_command(self) -> None:
        """Test that benchmark help command works."""
        cmd = ["python", "-m", "veeksha.benchmark", "-h"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, "Help command failed"
        assert "usage:" in result.stdout.lower(), "Help output missing usage"
        # assert "--client-config-model" in result.stdout, "Help missing expected arguments"

    def test_capacity_search_help_command(self) -> None:
        """Test that capacity search help command works."""
        cmd = ["python", "-m", "veeksha.capacity_search", "-h"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, "Help command failed"
        assert "usage:" in result.stdout.lower(), "Help output missing usage"
        # assert "--slos" in result.stdout, "Help missing SLOs argument"
        assert "--max-iterations" in result.stdout, "Help missing max iterations argument"