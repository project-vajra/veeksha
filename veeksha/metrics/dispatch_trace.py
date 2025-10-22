import csv
import os
import sys
from typing import Optional


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _lock_file(f):
    if os.name == "posix":
        try:
            import fcntl  # type: ignore

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass


def _unlock_file(f):
    if os.name == "posix":
        try:
            import fcntl  # type: ignore

            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass


def append_dispatch_trace(
    output_dir: str,
    request_id: Optional[int],
    session_id: Optional[int],
    ready_timestamp: Optional[float],
    dispatch_timestamp: float,
) -> None:
    """Append a single row to the dispatch trace CSV.

    The CSV schema is: request_id, session_id, ready_timestamp, dispatch_timestamp
    Timestamps are wall-clock seconds (time.time()).
    """
    if not output_dir:
        return

    trace_path = os.path.join(output_dir, "dispatch_trace.csv")
    _ensure_parent_dir(trace_path)

    file_exists = os.path.exists(trace_path) and os.path.getsize(trace_path) > 0

    try:
        with open(trace_path, mode="a", newline="", encoding="utf-8") as f:
            _lock_file(f)
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "request_id",
                        "session_id",
                        "ready_timestamp",
                        "dispatch_timestamp",
                    ]
                )
            writer.writerow(
                [
                    request_id if request_id is not None else "",
                    session_id if session_id is not None else "",
                    f"{ready_timestamp:.6f}" if ready_timestamp is not None else "",
                    f"{dispatch_timestamp:.6f}",
                ]
            )
            f.flush()
    except Exception:
        # Do not raise from metrics path; best-effort only
        return


