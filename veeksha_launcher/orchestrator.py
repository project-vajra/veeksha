"""Orchestrator that runs a sweep under an optional managed engine."""

from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml

from veeksha.sweeps import planner as sweep_planner
from veeksha_launcher.config import LauncherConfig
from veeksha_launcher.engines import (
    BaseEngineRunner,
    EngineRestartLimitExceeded,
    create_engine_runner,
)
from veeksha_launcher.events import (
    attempt_log_name,
    console_message,
    run_event_payload,
    sweep_plan_payload,
)
from veeksha_launcher.processes import ProcessTerminator
from veeksha_launcher.progress import (
    BenchmarkProgressReader,
    BenchmarkRequestProgress,
    LauncherProgressReporter,
    request_progress_payload,
)

_BENCHMARK_PROGRESS_INTERVAL_SECONDS = 30.0


@dataclass(frozen=True)
class BenchmarkAttemptResult:
    success: bool
    reason: str
    returncode: Optional[int] = None
    request_progress: Optional[BenchmarkRequestProgress] = None


class LauncherOrchestrator:
    def __init__(
        self,
        config: LauncherConfig,
        *,
        engine_runner: Optional[BaseEngineRunner] = None,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
        terminator: Optional[ProcessTerminator] = None,
        progress_reporter: Optional[LauncherProgressReporter] = None,
    ):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.engine_output_dir = self.output_dir / "engine"
        self.benchmark_log_dir = self.output_dir / "benchmark_logs"
        self.generated_config_dir = self.output_dir / "generated_configs"
        self._engine = engine_runner
        self._popen_factory = popen_factory
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn
        self._terminator = terminator or ProcessTerminator()
        self._event_file = None
        self._progress = progress_reporter or LauncherProgressReporter()

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_log_dir.mkdir(parents=True, exist_ok=True)
        self._persist_launcher_config()

        events_path = self.output_dir / "launcher_events.jsonl"
        with ExitStack() as stack:
            self._event_file = stack.enter_context(
                events_path.open("a", encoding="utf-8")
            )
            stack.callback(self._clear_event_file)
            stack.callback(self._progress.close)

            self._record_event(
                "launcher_start",
                output_dir=str(self.output_dir),
                events_file=str(events_path),
                benchmark_log_dir=str(self.benchmark_log_dir),
            )

            engine = self._resolve_engine()
            plan = sweep_planner.build_sweep_plan_from_config(
                self.config.sweep,
                client_api_base=engine.get_api_base() if engine is not None else None,
                tmp_parent=self.generated_config_dir,
            )
            self._record_event(
                "sweep_plan_ready",
                **sweep_plan_payload(plan, self.generated_config_dir),
            )

            if engine is not None:
                self._record_event("engine_start", **self._engine_event_payload(engine))
                engine.start()
                stack.callback(self._stop_engine, engine)
                self._record_event("engine_ready", **self._engine_event_payload(engine))
            else:
                self._record_event("engine_unmanaged")
            for run in plan.runs:
                self._run_descriptor(engine, plan.spec, run)
            self._record_event("sweep_complete", runs=len(plan.runs))

    def _clear_event_file(self) -> None:
        self._event_file = None

    def _stop_engine(self, engine: BaseEngineRunner) -> None:
        self._record_event("engine_stop", **self._engine_event_payload(engine))
        engine.stop()
        self._record_event("engine_stopped", **self._engine_event_payload(engine))

    def _resolve_engine(self) -> Optional[BaseEngineRunner]:
        if self._engine is not None:
            return self._engine
        if self.config.engine is None:
            return None
        self.engine_output_dir.mkdir(parents=True, exist_ok=True)
        return create_engine_runner(self.config.engine, self.engine_output_dir)

    def _engine_event_payload(self, engine: BaseEngineRunner) -> dict[str, str]:
        payload = {
            "runner": engine.__class__.__name__,
            "api_base": engine.get_api_base(),
            "health_url": engine.config.health_check_url,
            "engine_log_dir": str(engine.output_dir),
        }
        container_name = getattr(engine, "container_name", None)
        if isinstance(container_name, str):
            payload["container"] = container_name
        return payload

    def _run_descriptor(
        self,
        engine: Optional[BaseEngineRunner],
        spec: sweep_planner.SweepSpec,
        run: sweep_planner.SweepRunDescriptor,
    ) -> None:
        if engine is not None:
            engine.reset_restart_budget()
        max_attempts = self.config.retry.max_attempts_per_run
        attempt = 1
        while attempt <= max_attempts:
            if engine is not None and not self._ensure_engine_ready(engine):
                self._record_run_exhausted(
                    spec,
                    run,
                    f"sweep run {run.run_index}/{run.run_count} aborted: "
                    "engine restart budget exhausted",
                )
                return
            stdout_path, stderr_path = self._attempt_log_paths(run.run_index, attempt)
            start_payload = run_event_payload(
                spec, run, completed_runs=run.run_index - 1
            )
            start_payload.update(
                attempt=attempt,
                command=run.command,
                config=str(run.run_config),
                output_dir=run.output_dir,
                stdout_log=str(stdout_path),
                stderr_log=str(stderr_path),
                timeout_seconds=run.timeout_seconds,
                **request_progress_payload(
                    BenchmarkRequestProgress(
                        completed_requests=0,
                        total_requests=(
                            run.max_sessions if run.max_sessions > 0 else None
                        ),
                    )
                ),
            )
            self._record_event("benchmark_attempt_start", **start_payload)
            result = self._run_benchmark_attempt(
                engine, spec, run, attempt, stdout_path, stderr_path
            )
            if result.success:
                success_payload = run_event_payload(
                    spec, run, completed_runs=run.run_index
                )
                success_payload.update(attempt=attempt, returncode=result.returncode)
                if result.request_progress is not None:
                    success_payload.update(
                        request_progress_payload(result.request_progress)
                    )
                self._record_event("benchmark_attempt_success", **success_payload)
                if run.run_index < run.run_count and self._cooldown_seconds > 0:
                    cooldown_payload = run_event_payload(
                        spec, run, completed_runs=run.run_index
                    )
                    cooldown_payload.update(
                        seconds=self._cooldown_seconds,
                        next_run_index=run.run_index + 1,
                    )
                    self._record_event("cooldown_start", **cooldown_payload)
                    self._sleep(self._cooldown_seconds)
                return

            failure_payload = run_event_payload(
                spec, run, completed_runs=run.run_index - 1
            )
            failure_payload.update(
                attempt=attempt,
                reason=result.reason,
                returncode=result.returncode,
            )
            if result.request_progress is not None:
                failure_payload.update(
                    request_progress_payload(result.request_progress)
                )
            self._record_event("benchmark_attempt_failed", **failure_payload)
            if attempt >= max_attempts:
                self._record_run_exhausted(
                    spec,
                    run,
                    f"sweep run {run.run_index}/{run.run_count} failed after "
                    f"{max_attempts} attempts: {result.reason}",
                )
                return

            if engine is not None and (
                result.reason.startswith("engine_")
                or self.config.retry.restart_engine_before_retry
            ):
                if not self._restart_engine(engine, result.reason):
                    self._record_run_exhausted(
                        spec,
                        run,
                        f"sweep run {run.run_index}/{run.run_count} aborted: "
                        "engine restart budget exhausted",
                    )
                    return
            attempt += 1

    def _record_run_exhausted(
        self,
        spec: sweep_planner.SweepSpec,
        run: sweep_planner.SweepRunDescriptor,
        message: str,
    ) -> None:
        payload = run_event_payload(spec, run, completed_runs=run.run_index - 1)
        payload.update(message=message)
        self._record_event("benchmark_attempts_exhausted", **payload)
        if self.config.retry.fail_sweep_after_exhausted_retries:
            raise RuntimeError(message)

    @property
    def _cooldown_seconds(self) -> float:
        return float(self.config.sweep.cooldown_seconds or 0)

    def _ensure_engine_ready(self, engine: BaseEngineRunner) -> bool:
        """Ensure the engine is alive and healthy, restarting it if needed.

        Returns False only when a required restart hit the restart budget, so
        the caller can fail the run gracefully instead of crashing the sweep.
        """
        if not engine.is_alive():
            return self._restart_engine(engine, "engine_not_alive_before_run")
        if not engine.health_check():
            return self._restart_engine(engine, "engine_unhealthy_before_run")
        return True

    def _restart_engine(self, engine: BaseEngineRunner, reason: str) -> bool:
        """Restart the engine, emitting events. Return False if the budget is exhausted."""
        self._record_event(
            "engine_restart", reason=reason, **self._engine_event_payload(engine)
        )
        try:
            engine.restart()
        except EngineRestartLimitExceeded as exc:
            self._record_event(
                "engine_restart_exhausted",
                reason=reason,
                message=str(exc),
                **self._engine_event_payload(engine),
            )
            return False
        self._record_event("engine_ready", **self._engine_event_payload(engine))
        return True

    def _run_benchmark_attempt(
        self,
        engine: Optional[BaseEngineRunner],
        spec: sweep_planner.SweepSpec,
        run: sweep_planner.SweepRunDescriptor,
        attempt: int,
        stdout_path: Path,
        stderr_path: Path,
    ) -> BenchmarkAttemptResult:
        progress_path = self.benchmark_log_dir / attempt_log_name(
            run.run_index, attempt, "progress.json"
        )
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            process = self._popen_factory(
                run.command,
                cwd=str(sweep_planner.REPO_ROOT),
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                start_new_session=True,
                env={**os.environ, "VEEKSHA_PROGRESS_FILE": str(progress_path)},
            )
            started_at = self._monotonic()
            progress_reader = BenchmarkProgressReader(progress_path, run.max_sessions)
            if engine is None:
                last_health_check = 0.0
            else:
                last_health_check = started_at - engine.config.health_check_interval
            last_progress_report = started_at
            while True:
                returncode = process.poll()
                if returncode is not None:
                    return BenchmarkAttemptResult(
                        success=returncode == 0,
                        reason="completed" if returncode == 0 else "benchmark_failed",
                        returncode=returncode,
                        request_progress=progress_reader.read(),
                    )

                if engine is not None and not engine.is_alive():
                    self._terminate_benchmark_process(process)
                    return BenchmarkAttemptResult(
                        success=False,
                        reason="engine_exited",
                        request_progress=progress_reader.read(),
                    )

                poll_interval = 1.0
                now = self._monotonic()
                request_progress = progress_reader.read()
                self._progress.update_attempt_requests(
                    request_progress.completed_requests, request_progress.total_requests
                )
                if engine is not None:
                    poll_interval = min(1.0, engine.config.health_check_interval)
                    if now - last_health_check >= engine.config.health_check_interval:
                        last_health_check = now
                        if not engine.health_check():
                            self._terminate_benchmark_process(process)
                            return BenchmarkAttemptResult(
                                success=False,
                                reason="engine_unhealthy",
                                request_progress=progress_reader.read(),
                            )

                if now - last_progress_report >= _BENCHMARK_PROGRESS_INTERVAL_SECONDS:
                    last_progress_report = now
                    self._record_benchmark_progress(
                        spec,
                        run,
                        attempt,
                        elapsed_seconds=now - started_at,
                        request_progress=request_progress,
                    )

                self._sleep(poll_interval)

    def _terminate_benchmark_process(self, process: subprocess.Popen) -> None:
        self._terminator.terminate(process)

    def _record_benchmark_progress(
        self,
        spec: sweep_planner.SweepSpec,
        run: sweep_planner.SweepRunDescriptor,
        attempt: int,
        *,
        elapsed_seconds: float,
        request_progress: BenchmarkRequestProgress,
    ) -> None:
        payload = run_event_payload(spec, run, completed_runs=run.run_index - 1)
        payload.update(
            attempt=attempt,
            elapsed_seconds=round(elapsed_seconds, 1),
            timeout_seconds=run.timeout_seconds,
            **request_progress_payload(request_progress),
        )
        self._record_event("benchmark_attempt_progress", **payload)

    def _attempt_log_paths(self, run_index: int, attempt: int) -> tuple[Path, Path]:
        return (
            self.benchmark_log_dir / attempt_log_name(run_index, attempt, "stdout.log"),
            self.benchmark_log_dir / attempt_log_name(run_index, attempt, "stderr.log"),
        )

    def _persist_launcher_config(self) -> None:
        path = self.output_dir / "launcher_config.yml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.config.to_dict(), f, sort_keys=False)

    def _record_event(self, event: str, **payload) -> None:
        if self._event_file is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        self._event_file.write(json.dumps(record, sort_keys=True) + "\n")
        self._event_file.flush()
        self._progress.handle_event(event, payload)
        self._print_event(event, payload)

    def _print_event(self, event: str, payload: dict) -> None:
        message = console_message(event, payload)
        if message is not None:
            self._progress.write(message)
