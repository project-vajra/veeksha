"""Unit tests for CLI commands."""

import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from types import ModuleType
from typing import Iterator

import pytest

import veeksha.__main__ as veeksha_main
import veeksha.cli.benchmarks as benchmark_cli
import veeksha.cli.commands as cli_commands
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.endpoint import EndpointConfig
from veeksha.config.server import BaseServerConfig
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

    def test_benchmark_cli_expands_and_groups_managed_server_lifecycles(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Parse, expand, group, and run one lifecycle per server config."""
        deploy_config = tmp_path / "deploy.yml"
        deploy_config.write_text("model: fixture\n", encoding="utf-8")
        output_dir = tmp_path / "benchmark_output"
        config_path = tmp_path / "managed_sweep.yml"
        config_path.write_text(
            dedent(
                f"""
                output_dir: {output_dir}
                server:
                  type: vllm
                  image: vllm-omni:test
                  hf_model: meta/demo-model
                  deploy_config: {deploy_config}
                  bootstrap: ""
                  port: !expand [8101, 8102]
                traffic_scheduler:
                  type: rate
                  interval_generator:
                    type: poisson
                    arrival_rate: !expand [1.0, 2.0]
                runtime:
                  max_sessions: 1
                  benchmark_timeout: 1
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        lifecycle_events: list[tuple[str, int]] = []
        endpoint_ids_by_port: dict[int, set[int]] = defaultdict(set)
        rates_by_port: dict[int, set[float]] = defaultdict(set)
        server_output_dirs: list[str] = []
        summary_run_dirs: list[str] = []
        num_runs = 0

        @contextmanager
        def fake_managed_server(
            config: BaseServerConfig,
            output_dir: str,
        ) -> Iterator[dict[str, object]]:
            assert isinstance(config.startup_timeout, float)
            assert config.startup_timeout == 1.5
            lifecycle_events.append(("start", config.port))
            server_output_dirs.append(output_dir)
            try:
                yield {"endpoint": config.get_endpoint()}
            finally:
                lifecycle_events.append(("stop", config.port))

        def fake_run_benchmark_with_endpoint(
            config: BenchmarkConfig,
            endpoint: EndpointConfig,
        ) -> None:
            nonlocal num_runs
            run_dir = Path(config.output_dir) / f"run_{num_runs:02d}"
            run_dir.mkdir(parents=True)
            object.__setattr__(config, "output_dir", str(run_dir))
            num_runs += 1
            endpoint_ids_by_port[endpoint.port].add(id(endpoint))
            rates_by_port[endpoint.port].add(
                config.traffic_scheduler.interval_generator.arrival_rate
            )

        def fake_write_sweep_summary(
            _sweep_dir: str,
            run_dirs: list[str],
        ) -> None:
            summary_run_dirs.extend(run_dirs)

        monkeypatch.setattr(
            benchmark_cli,
            "managed_server",
            fake_managed_server,
        )
        monkeypatch.setattr(
            benchmark_cli,
            "run_benchmark_with_endpoint",
            fake_run_benchmark_with_endpoint,
        )
        monkeypatch.setattr(
            benchmark_cli,
            "write_sweep_summary",
            fake_write_sweep_summary,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "veeksha",
                "benchmark",
                "--config",
                str(config_path),
                "--server.startup_timeout",
                "1.5",
            ],
        )

        cli_commands.main()

        assert lifecycle_events == [
            ("start", 8101),
            ("stop", 8101),
            ("start", 8102),
            ("stop", 8102),
        ]
        assert {port: len(ids) for port, ids in endpoint_ids_by_port.items()} == {
            8101: 1,
            8102: 1,
        }
        assert rates_by_port == {8101: {1.0, 2.0}, 8102: {1.0, 2.0}}
        assert len(summary_run_dirs) == 4
        assert len(set(summary_run_dirs)) == 4
        assert [Path(path).name for path in server_output_dirs] == [
            "managed_server_01",
            "managed_server_02",
        ]
