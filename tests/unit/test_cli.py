"""Unit tests for CLI commands."""

import subprocess
import sys
from types import ModuleType

import pytest

import veeksha.__main__ as veeksha_main
from veeksha.version import __version__


@pytest.mark.unit
class TestCLI:
    """Test CLI commands work correctly."""

    def test_top_level_version_flag(self) -> None:
        """Test that the top-level version flag works."""
        cmd = ["python", "-m", "veeksha", "--version"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, "Version command failed"
        assert (
            result.stdout.strip() == f"veeksha {__version__}"
        ), "Version output mismatch"

    def test_module_entrypoint_requires_free_threaded_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the shared CLI entrypoint rejects non-free-threaded Python."""
        monkeypatch.setattr(
            veeksha_main.sysconfig,
            "get_config_var",
            lambda name: False if name == "Py_GIL_DISABLED" else None,
        )

        with pytest.raises(SystemExit, match="veeksha requires free-threaded Python"):
            veeksha_main.main()

    def test_module_entrypoint_invokes_cli_main(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the shared CLI entrypoint dispatches to the CLI main."""
        called = False

        def fake_cli_main() -> None:
            nonlocal called
            called = True

        fake_commands = ModuleType("veeksha.cli.commands")
        fake_commands.main = fake_cli_main

        monkeypatch.setattr(
            veeksha_main.sysconfig,
            "get_config_var",
            lambda name: True if name == "Py_GIL_DISABLED" else None,
        )
        monkeypatch.setitem(sys.modules, "veeksha.cli.commands", fake_commands)

        veeksha_main.main()

        assert called, "CLI main was not invoked"

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
        assert (
            "--max_iterations" in result.stdout
        ), "Help missing max iterations argument"
