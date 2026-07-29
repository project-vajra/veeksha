"""Managed server lifecycle runners for Veeksha benchmarks."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Generic, Optional, TypeVar

import requests

from veeksha.config.server import (
    ManagedServerConfig,
    SglangServerConfig,
    VajraServerConfig,
    VllmServerConfig,
)
from veeksha.orchestration.processes import ProcessTerminator
from veeksha.orchestration.server_manager import BaseServerManager

_ENGINE_DETAILS_FILENAME = "engine_details.json"

ManagedConfigT = TypeVar("ManagedConfigT", bound=ManagedServerConfig)
DockerConfigT = TypeVar(
    "DockerConfigT",
    bound=VllmServerConfig | SglangServerConfig,
)


class EngineError(RuntimeError):
    """Base engine lifecycle error."""


class BaseEngineRunner(
    BaseServerManager[ManagedConfigT],
    ABC,
    Generic[ManagedConfigT],
):
    def __init__(
        self,
        config: ManagedConfigT,
        output_dir: str | Path,
        *,
        terminator: Optional[ProcessTerminator] = None,
    ):
        super().__init__(config, output_dir=str(output_dir))
        self.output_dir = Path(output_dir)
        self._delete_log_file_on_cleanup = False
        self._terminator = terminator or ProcessTerminator()

    @property
    def is_running(self) -> bool:
        raise EngineError(
            "managed server lifecycle uses is_alive(); is_running is invalid"
        )

    def get_api_base(self) -> str:
        return self.get_endpoint().api_base

    def get_endpoint(self):
        return self.config.get_endpoint()

    def launch(self) -> tuple[bool, Optional[str]]:
        raise EngineError("managed server lifecycle uses start(); launch is invalid")

    def shutdown(self, force: bool = False) -> bool:
        raise EngineError("managed server lifecycle uses stop(); shutdown is invalid")

    def get_server_logs(self, lines: int = 50) -> tuple[str, str]:
        return self.tail_logs(lines), ""

    def _build_launch_command(self) -> list[str]:
        return []

    def _ensure_managed_port_available(self) -> None:
        if self._is_port_in_use():
            raise EngineError(
                f"port {self.config.port} on host {self.config.host!r} is already in use; "
                "refusing to attach managed server health checks to an existing listener"
            )

    @abstractmethod
    def start(self) -> None:
        """Start the engine and wait until ready."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the engine if it is running."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Return whether the underlying process/container is still running."""

    @abstractmethod
    def tail_logs(self, lines: int = 80) -> str:
        """Return recent engine logs for diagnostics."""

    def health_check(self) -> bool:
        try:
            response = requests.get(self.config.health_check_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def wait_for_ready(self, timeout: Optional[float] = None) -> bool:
        try:
            self._wait_for_ready_or_raise(timeout=timeout)
        except (EngineError, TimeoutError):
            return False
        return True

    def _wait_for_ready_or_raise(self, timeout: Optional[float] = None) -> None:
        startup_timeout = self.config.startup_timeout if timeout is None else timeout
        start = time.monotonic()
        while time.monotonic() - start < startup_timeout:
            if not self.is_alive():
                raise EngineError(
                    "engine exited before becoming ready\n" + self.tail_logs()
                )
            if self.health_check():
                return
            time.sleep(self.config.health_check_interval)
        raise TimeoutError(
            f"engine did not become ready within {startup_timeout}s\n"
            + self.tail_logs()
        )

    def _write_engine_details(self, payload: dict[str, object]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / _ENGINE_DETAILS_FILENAME
        record = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


class VajraSubprocessRunner(BaseEngineRunner[VajraServerConfig]):
    def __init__(
        self,
        config: VajraServerConfig,
        output_dir: str | Path,
        *,
        terminator: Optional[ProcessTerminator] = None,
    ):
        super().__init__(config, output_dir, terminator=terminator)
        self._process: Optional[subprocess.Popen] = None
        self._stdout_file: Optional[IO[str]] = None
        self._stderr_file: Optional[IO[str]] = None
        self._stdout_path: Optional[Path] = None
        self._stderr_path: Optional[Path] = None

    def start(self) -> None:
        if self.is_alive():
            return
        self._ensure_managed_port_available()
        allocation_success, allocation_error = self._ensure_gpu_allocation()
        if not allocation_success:
            raise EngineError(allocation_error or "failed to allocate GPUs")

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.start_count += 1
            self._write_engine_details(self._build_git_details())
            self._stdout_path = self.output_dir / f"vajra_stdout_{self.start_count}.log"
            self._stderr_path = self.output_dir / f"vajra_stderr_{self.start_count}.log"
            self._stdout_file = self._stdout_path.open("w", encoding="utf-8")
            self._stderr_file = self._stderr_path.open("w", encoding="utf-8")
            self._process = subprocess.Popen(
                list(self.config.command),
                cwd=str(self._setup_dir()),
                stdout=self._stdout_file,
                stderr=self._stderr_file,
                start_new_session=True,
                env=self._build_env(),
                text=True,
            )
            self._wait_for_ready_or_raise()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        try:
            if self._process is not None:
                self._terminator.terminate(self._process)
        finally:
            self._close_logs()
            self._process = None
            self._release_allocated_resources()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def tail_logs(self, lines: int = 80) -> str:
        chunks = []
        for label, path in (
            ("stderr", self._stderr_path),
            ("stdout", self._stdout_path),
        ):
            if path is None or not path.exists():
                continue
            try:
                content = path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                chunks.append(f"[{label}]\n" + "\n".join(content[-lines:]))
            except OSError as exc:
                chunks.append(f"[{label}] failed to read {path}: {exc}")
        return "\n".join(chunks)

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.config.env or {})
        setup_dir = str(self._setup_dir())
        pythonpath = env.get("PYTHONPATH")
        paths = [
            path
            for path in (pythonpath.split(os.pathsep) if pythonpath else [])
            if path and path != setup_dir
        ]
        env["PYTHONPATH"] = os.pathsep.join([setup_dir, *paths])
        if self.config.gpu_ids is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(gpu) for gpu in self.config.gpu_ids
            )
        return env

    def _setup_dir(self) -> Path:
        if not self.config.setup_dir:
            raise EngineError("vajra server.setup_dir is required")
        return Path(self.config.setup_dir).expanduser()

    def _close_logs(self) -> None:
        for file_obj in (self._stdout_file, self._stderr_file):
            if file_obj is not None and not file_obj.closed:
                file_obj.close()
        self._stdout_file = None
        self._stderr_file = None

    def _build_git_details(self) -> dict[str, object]:
        setup_dir = self._setup_dir()
        result = subprocess.run(
            ["git", "-C", str(setup_dir), "log", "-1", "--format=%H"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise EngineError(
                "failed to read latest Vajra git commit with git log "
                f"in {setup_dir}: {result.stderr.strip() or result.stdout.strip()}"
            )
        git_commit_id = result.stdout.strip()
        if not git_commit_id:
            raise EngineError(f"git log returned no commit ID in {setup_dir}")
        return {
            "engine_type": self.config.type,
            "source": "git",
            "git_commit_id": git_commit_id,
            "git_source_dir": str(setup_dir),
            "git_command": "git log -1 --format=%H",
        }


class VllmOmniDockerRunner(
    BaseEngineRunner[DockerConfigT],
    Generic[DockerConfigT],
):
    def __init__(
        self,
        config: DockerConfigT,
        output_dir: str | Path,
        *,
        terminator: Optional[ProcessTerminator] = None,
    ):
        super().__init__(config, output_dir, terminator=terminator)
        self._container_name = _unique_container_name(config.container_name_prefix)
        self._container_id: Optional[str] = None
        self._log_process: Optional[subprocess.Popen] = None
        self._log_file: Optional[IO[str]] = None

    @property
    def container_name(self) -> str:
        return self._container_name

    def start(self) -> None:
        if self.is_alive():
            return
        self._ensure_managed_port_available()
        allocation_success, allocation_error = self._ensure_gpu_allocation()
        if not allocation_success:
            raise EngineError(allocation_error or "failed to allocate GPUs")

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.start_count += 1
            cmd = self._build_docker_run_cmd()
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise EngineError(
                    "docker run failed "
                    f"(rc={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
            self._container_id = result.stdout.strip()
            self._write_engine_details(self._build_docker_details())
            self._start_log_streamer()
            self._wait_for_ready_or_raise()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        try:
            self._stop_log_streamer()
            if self._container_id is None and not self.is_alive():
                return
            try:
                subprocess.run(
                    ["docker", "stop", self._container_name],
                    capture_output=True,
                    timeout=60,
                )
            except subprocess.TimeoutExpired:
                pass
            subprocess.run(
                ["docker", "rm", "-f", self._container_name], capture_output=True
            )
            self._container_id = None
        finally:
            self._release_allocated_resources()

    def is_alive(self) -> bool:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{.State.Running}}",
                self._container_name,
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() == "true"

    def tail_logs(self, lines: int = 80) -> str:
        result = subprocess.run(
            [
                "docker",
                "logs",
                "--tail",
                str(lines),
                self._container_name,
            ],
            capture_output=True,
            text=True,
        )
        return f"{result.stdout}\n{result.stderr}".strip()

    def _build_docker_run_cmd(self) -> list[str]:
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self._container_name,
        ]
        if self.config.docker_runtime:
            cmd.extend(["--runtime", self.config.docker_runtime])
        cmd.extend(["--gpus", self._docker_gpu_arg()])
        if self.config.ipc_mode:
            cmd.append(f"--ipc={self.config.ipc_mode}")
        cmd.extend(self._extra_run_args())
        cmd.extend(
            [
                "-p",
                f"{self.config.port}:{self.config.resolved_container_port}",
                "--init",
            ]
        )
        for volume in [
            *self.config.volumes,
            *self._extra_volumes(),
            self._deploy_config_volume(),
        ]:
            cmd.extend(["-v", volume])
        config_env = self.config.env or {}
        for key, value in config_env.items():
            cmd.extend(["-e", f"{key}={value}"])
        for key in self.config.pass_env:
            if key not in config_env:
                if key not in os.environ:
                    raise EngineError(
                        f"required environment variable is not set: {key}"
                    )
                cmd.extend(["-e", key])
        cmd.append(self.config.image)
        cmd.extend(self._build_server_cmd())
        return cmd

    def _docker_gpu_arg(self) -> str:
        if self.config.docker_gpus:
            return self.config.docker_gpus
        if self.config.gpu_ids is not None:
            return "device=" + ",".join(str(gpu) for gpu in self.config.gpu_ids)
        return "all"

    def _deploy_config_volume(self) -> str:
        deploy_config = Path(self.config.deploy_config)
        if not deploy_config.is_file():
            raise EngineError(f"deploy config not found: {deploy_config}")
        return f"{deploy_config}:{self.config.resolved_container_deploy_config}:ro"

    def _build_server_cmd(self) -> list[str]:
        # Base command is vLLM-specific; SglangOmniDockerRunner overrides this.
        assert isinstance(self.config, VllmServerConfig)
        serve = [
            "vllm",
            "serve",
            self.config.hf_model,
            "--omni",
            "--port",
            str(self.config.resolved_container_port),
            "--deploy-config",
            self.config.resolved_container_deploy_config,
            *self.config.engine_args,
        ]
        if not self.config.uses_bootstrap:
            return serve
        serve_line = "exec " + " ".join(shlex.quote(part) for part in serve)
        script = self.config.resolved_bootstrap + "\n" + serve_line
        return ["bash", "-lc", script]

    def _extra_run_args(self) -> list[str]:
        """Extra ``docker run`` flags inserted before the port mapping."""
        return []

    def _extra_volumes(self) -> list[str]:
        """Extra ``-v`` mounts beyond ``volumes`` and the deploy config."""
        return []

    def _log_prefix(self) -> str:
        return "vllm_docker"

    def _start_log_streamer(self) -> None:
        self._log_file = (
            self.output_dir / f"{self._log_prefix()}_{self.start_count}.log"
        ).open("w", encoding="utf-8")
        self._log_process = subprocess.Popen(
            ["docker", "logs", "-f", self._container_name],
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    def _stop_log_streamer(self) -> None:
        if self._log_process is not None:
            self._terminator.terminate(self._log_process)
            self._log_process = None
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.close()
        self._log_file = None

    def _build_docker_details(self) -> dict[str, object]:
        if self._container_id is None:
            raise EngineError("cannot record Docker details before container start")
        return {
            "engine_type": self.config.type,
            "source": "docker",
            "docker_image": self.config.image,
            "docker_image_hash": self._docker_inspect_value(
                "{{.Image}}", self._container_id
            ),
            "container_id": self._container_id,
            "container_name": self._container_name,
        }

    def _docker_inspect_value(self, format_expr: str, target: str) -> str:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                format_expr,
                target,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise EngineError(
                "docker inspect failed "
                f"(target={target}, rc={result.returncode}): {result.stderr.strip()}"
            )
        value = result.stdout.strip()
        if not value:
            raise EngineError(f"docker inspect returned no value for {target}")
        return value


class SglangOmniDockerRunner(VllmOmniDockerRunner[SglangServerConfig]):
    """Runs sglang-omni's OpenAI-compatible server in a Docker container.

    Differences from the vLLM Omni runner:

    - The container command is ``sgl-omni serve --model-path ... --config ...``
      (``--config``, not ``--deploy-config``; no ``--omni`` flag).
    - The stock ``frankleeeee/sglang-omni:dev`` image only ships prerequisites,
      so the command is wrapped in a ``bash`` bootstrap that installs
      sglang-omni from the mounted source checkout before serving (configurable;
      set ``engine.bootstrap`` to an empty string for a pre-baked image).
    - Adds ``--shm-size`` (sglang-omni uses a shared-memory relay) and mounts the
      source checkout. Health is checked at ``/health``.
    """

    def __init__(
        self,
        config: SglangServerConfig,
        output_dir: str | Path,
        *,
        terminator: Optional[ProcessTerminator] = None,
    ):
        super().__init__(config, output_dir, terminator=terminator)

    def _log_prefix(self) -> str:
        return "sglang_docker"

    def _extra_run_args(self) -> list[str]:
        args: list[str] = []
        if self.config.shm_size:
            args.extend(["--shm-size", self.config.shm_size])
        args.extend(self.config.docker_run_args)
        return args

    def _extra_volumes(self) -> list[str]:
        if self.config.uses_bootstrap and self.config.source_dir:
            return [f"{self.config.source_dir}:{self.config.container_source_dir}"]
        return []

    def _build_server_cmd(self) -> list[str]:
        serve = [
            "sgl-omni",
            "serve",
            "--model-path",
            self.config.model_path,
            "--config",
            self.config.resolved_container_deploy_config,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.config.resolved_container_port),
        ]
        if self.config.model_name:
            serve.extend(["--model-name", self.config.model_name])
        serve.extend(self.config.engine_args)
        if not self.config.uses_bootstrap:
            return serve
        serve_line = "exec " + " ".join(shlex.quote(part) for part in serve)
        script = self.config.resolved_bootstrap + "\n" + serve_line
        return ["bash", "-lc", script]


def _unique_container_name(prefix: str) -> str:
    return f"{prefix}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
