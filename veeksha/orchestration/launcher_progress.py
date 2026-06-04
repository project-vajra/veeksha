"""Progress parsing and rendering for orchestrated launcher runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm


@dataclass(frozen=True)
class BenchmarkRequestProgress:
    completed_requests: int
    total_requests: Optional[int] = None


class BenchmarkProgressReader:
    """Read benchmark progress from the JSON file the benchmark publishes.

    The launcher points the benchmark subprocess at this file via the
    ``VEEKSHA_PROGRESS_FILE`` environment variable; the benchmark writes
    ``{"completed": int, "total": int | null}`` to it atomically. This is a
    dedicated machine-readable channel, so there is no console scraping and no
    coupling to the benchmark's progress-bar formatting.
    """

    def __init__(self, progress_path: Path, fallback_total_requests: int):
        self._progress_path = progress_path
        total_requests = positive_int_or_none(fallback_total_requests)
        self._progress = BenchmarkRequestProgress(
            completed_requests=0, total_requests=total_requests
        )

    def read(self) -> BenchmarkRequestProgress:
        try:
            raw = self._progress_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (FileNotFoundError, OSError, ValueError):
            # Not written yet, transiently unreadable, or mid-write: keep last value.
            return self._progress

        completed = data.get("completed")
        if not isinstance(completed, int) or isinstance(completed, bool):
            return self._progress

        total = positive_int_or_none(data.get("total"))
        if total is None:
            total = self._progress.total_requests
        self._progress = BenchmarkRequestProgress(
            completed_requests=max(completed, 0), total_requests=total
        )
        return self._progress


def request_progress_payload(progress: BenchmarkRequestProgress) -> dict:
    payload = {
        "requests_completed": progress.completed_requests,
        "request_total": progress.total_requests,
    }
    if progress.total_requests is None:
        payload["request_progress_pct"] = None
    else:
        payload["request_progress_pct"] = progress_percentage(
            progress.completed_requests, progress.total_requests
        )
    return payload


def positive_int_or_none(value: object) -> Optional[int]:
    if not isinstance(value, int) or value <= 0:
        return None
    return value


def progress_percentage(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 100.0
    return round(min(max(numerator / denominator * 100, 0.0), 100.0), 1)


class LauncherProgressReporter:
    """TQDM-backed console progress for orchestrated launcher runs."""

    def __init__(self) -> None:
        self._sweep_bar: Optional[Any] = None
        self._attempt_bar: Optional[Any] = None

    def handle_event(self, event: str, payload: dict) -> None:
        if event == "sweep_plan_ready":
            self._start_sweep(payload)
            return
        if event == "benchmark_attempt_start":
            self._start_attempt(payload)
            return
        if event == "benchmark_attempt_progress":
            self.update_attempt_requests(
                int(payload["requests_completed"]),
                positive_int_or_none(payload.get("request_total")),
            )
            return
        if event == "benchmark_attempt_success":
            self.update_attempt_requests(
                int(payload["requests_completed"]),
                positive_int_or_none(payload.get("request_total")),
            )
            self._close_attempt()
            self._set_sweep_completed(int(payload["completed_runs"]))
            return
        if event == "benchmark_attempt_failed":
            self.update_attempt_requests(
                int(payload["requests_completed"]),
                positive_int_or_none(payload.get("request_total")),
            )
            self._close_attempt()
            return
        if event == "benchmark_attempts_exhausted":
            self._close_attempt()
            self._set_sweep_completed(int(payload["run_index"]))
            return
        if event == "sweep_complete":
            self._finish_sweep()

    def update_attempt_requests(
        self, completed_requests: int, total_requests: Optional[int]
    ) -> None:
        if self._attempt_bar is None:
            return
        bounded_completed = max(completed_requests, 0)
        if total_requests is not None:
            self._attempt_bar.total = total_requests
            bounded_completed = min(bounded_completed, total_requests)
            self._attempt_bar.n = bounded_completed
            self._attempt_bar.set_postfix_str(
                f"requests={bounded_completed}/{total_requests}"
            )
        else:
            previous_completed = int(self._attempt_bar.n or 0)
            if bounded_completed >= previous_completed:
                self._attempt_bar.update(bounded_completed - previous_completed)
            else:
                self._attempt_bar.n = bounded_completed
            self._attempt_bar.set_postfix_str(f"requests={bounded_completed}")
        self._attempt_bar.refresh()

    def write(self, message: str) -> None:
        tqdm.write(f"[veeksha-launcher] {message}")

    def close(self) -> None:
        self._close_attempt()
        if self._sweep_bar is not None:
            self._sweep_bar.close()
            self._sweep_bar = None

    def _start_sweep(self, payload: dict) -> None:
        self.close()
        total_runs = int(payload["runs"])
        self._sweep_bar = tqdm(
            total=total_runs,
            desc=f"sweep {payload['sweep_type']} {payload['engine']} {payload['model']}",
            unit="run",
            dynamic_ncols=True,
            position=0,
        )
        self._set_sweep_completed(0)

    def _start_attempt(self, payload: dict) -> None:
        self._close_attempt()
        total_requests = positive_int_or_none(payload.get("request_total"))
        self._attempt_bar = tqdm(
            total=total_requests,
            desc=(
                f"run {payload['run_index']}/{payload['run_count']} "
                f"attempt {payload['attempt']}"
            ),
            unit="req",
            dynamic_ncols=True,
            position=1,
            leave=False,
        )
        self.update_attempt_requests(0, total_requests)

    def _close_attempt(self) -> None:
        if self._attempt_bar is not None:
            self._attempt_bar.close()
            self._attempt_bar = None

    def _set_sweep_completed(self, completed_runs: int) -> None:
        if self._sweep_bar is None:
            return
        total = int(self._sweep_bar.total or 0)
        self._sweep_bar.n = min(max(completed_runs, 0), total)
        self._sweep_bar.set_postfix_str(f"complete={self._sweep_bar.n}/{total}")
        self._sweep_bar.refresh()

    def _finish_sweep(self) -> None:
        if self._sweep_bar is not None:
            self._set_sweep_completed(int(self._sweep_bar.total or 0))
