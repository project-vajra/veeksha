#!/usr/bin/env python3
"""Analyze session dispatch rate from a dispatch_trace.csv file.

This script computes the rate at which sessions are dispatched (considering only
the first request of each session) and summarizes the delay between the ready
timestamp and the dispatch timestamp for those first requests.

It prints results to the console. If --input is not provided, it selects the
latest benchmark's dispatch trace from a specified root directory by parsing the
timestamp embedded in the run directory name.

Usage:
    python tools/analyze_session_dispatch.py \
        [--input /path/to/dispatch_trace.csv] [--root ./benchmark_results]
"""

import argparse
import csv
import os
from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, List, Optional, Tuple, Union
import re


@dataclass
class FirstSessionRecord:
    session_id: int
    request_id: Optional[int]
    ready_timestamp: Optional[float]
    dispatch_timestamp: float


def _parse_int(val: str) -> Optional[int]:
    try:
        return int(val)
    except Exception:
        return None


def _parse_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def _percentile(sorted_values: List[float], p: float) -> float:
    """Compute the pth percentile (0-100) from a sorted list (inclusive method)."""
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    n = len(sorted_values)
    # rank-based index (inclusive-style)
    idx = int(round((p / 100.0) * (n - 1)))
    return sorted_values[idx]


def load_first_requests_by_session(
    csv_path: str,
) -> Dict[int, FirstSessionRecord]:
    """Load dispatch_trace.csv and return first dispatched request per session.

    First is determined by the minimum dispatch_timestamp for each session_id.
    Rows without a valid session_id are ignored.
    """
    firsts: Dict[int, FirstSessionRecord] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sess_raw = row.get("session_id", "")
            session_id = _parse_int(sess_raw) if sess_raw != "" else None
            if session_id is None:
                continue

            req_raw = row.get("request_id", "")
            request_id = _parse_int(req_raw) if req_raw != "" else None

            ready_ts = _parse_float(row.get("ready_timestamp", ""))
            dispatch_ts_raw = row.get("dispatch_timestamp", "")
            dispatch_ts = _parse_float(dispatch_ts_raw)
            if dispatch_ts is None:
                continue

            existing = firsts.get(session_id)
            if existing is None or dispatch_ts < existing.dispatch_timestamp:
                firsts[session_id] = FirstSessionRecord(
                    session_id=session_id,
                    request_id=request_id,
                    ready_timestamp=ready_ts,
                    dispatch_timestamp=dispatch_ts,
                )

    return firsts


def compute_overall_rate(records: List[FirstSessionRecord]) -> float:
    if not records:
        return 0.0
    times = [r.dispatch_timestamp for r in records]
    span = max(times) - min(times)
    if span <= 0:
        return float("inf")
    return len(records) / span

def summarize_delays(records: List[FirstSessionRecord]) -> str:
    delays = [
        r.dispatch_timestamp - r.ready_timestamp
        for r in records
        if r.ready_timestamp is not None
    ]
    if not delays:
        return "No ready-to-dispatch delays available."
    delays_sorted = sorted(delays)
    return (
        "Delay stats (s): "
        f"count={len(delays_sorted)}, "
        f"mean={mean(delays_sorted):.6f}, "
        f"median={median(delays_sorted):.6f}, "
        f"p95={_percentile(delays_sorted, 95):.6f}, "
        f"p99={_percentile(delays_sorted, 99):.6f}, "
        f"max={delays_sorted[-1]:.6f}"
    )


def _extract_run_timestamp_key(dir_name: str) -> Optional[Tuple[int, int, int]]:
    """Extract a sortable timestamp key (YYYYMMDD, HHMMSS, suffix) from run dir name.

    Expected pattern: <model>-<hash>-YYYYMMDD-HHMMSS-<suffix>
    Returns None if pattern doesn't match.
    """
    m = re.search(r"-(\d{8})-(\d{6})-(\d+)$", dir_name)
    if not m:
        return None
    try:
        ymd = int(m.group(1))
        hms = int(m.group(2))
        suf = int(m.group(3))
        return (ymd, hms, suf)
    except Exception:
        return None


def find_latest_dispatch_trace(root_dir: str) -> Optional[str]:
    if not os.path.isdir(root_dir):
        return None
    candidates: List[Tuple[Tuple[int, int, int], str]] = []
    for name in os.listdir(root_dir):
        run_dir = os.path.join(root_dir, name)
        if not os.path.isdir(run_dir):
            continue
        trace_path = os.path.join(run_dir, "dispatch_trace.csv")
        if not os.path.exists(trace_path):
            continue
        key = _extract_run_timestamp_key(name)
        if key is None:
            # fallback to mtime-based key (roughly sortable)
            try:
                stat = os.stat(run_dir)
                # convert float to two-part int tuple for stable sort
                sec = int(stat.st_mtime)
                nsec = int((stat.st_mtime - sec) * 1e9)
                key = (sec, nsec, 0)
            except Exception:
                continue
        candidates.append((key, trace_path))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute session dispatch rate from dispatch_trace.csv"
    )
    parser.add_argument(
        "--input",
        required=False,
        help="Path to dispatch_trace.csv (if omitted, auto-detect latest in --root)",
    )
    parser.add_argument(
        "--root",
        required=False,
        default="benchmark_results",
        help="Root directory containing benchmark run subdirectories",
    )
    parser.add_argument(
        "--n",
        required=False,
        help=(
            "How many earliest sessions to include (int, 'all', or comma-separated list). "
            "Default: 50,100,200,all"
        ),
    )
    args = parser.parse_args()

    if args.input:
        input_path = os.path.abspath(args.input)
    else:
        input_path = find_latest_dispatch_trace(os.path.abspath(args.root)) or ""
    assert input_path and os.path.exists(input_path), (
        "Could not locate dispatch_trace.csv. Provide --input or set --root to a directory "
        "containing benchmark run subdirectories."
    )

    print(f"Using dispatch trace: {input_path}")

    firsts_by_session = load_first_requests_by_session(input_path)
    first_records = list(firsts_by_session.values())
    first_records.sort(key=lambda r: r.dispatch_timestamp)
    print(f"Sessions detected: {len(first_records)}")

    def _parse_n_arg(n_arg: Optional[str]) -> List[Union[int, str]]:
        if not n_arg:
            return [50, 100, 200, "all"]
        parts = [p.strip() for p in n_arg.split(",")]
        out: List[Union[int, str]] = []
        for p in parts:
            if p.lower() == "all":
                out.append("all")
            else:
                try:
                    out.append(int(p))
                except Exception:
                    pass
        seen = set()
        dedup: List[Union[int, str]] = []
        for v in out:
            key = ("all",) if v == "all" else ("int", int(v))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(v)
        return dedup or [50, 100, 200, "all"]

    strata = _parse_n_arg(args.n)
    # Collect series for plotting
    x_ns: List[int] = []
    y_rates: List[float] = []
    y_mean_delays: List[float] = []

    for s in strata:
        k = None if s == "all" else int(s)
        records = first_records if k is None else first_records[: max(0, k)]
        label = f"n={s}"
        overall_rate = compute_overall_rate(records)
        print(
            f"[{label}] sessions used: {len(records)} | session dispatch rate: {overall_rate:.6f} /s"
        )
        print(summarize_delays(records))
        print("--------------------------------")

        # Build plotting series
        n_value = len(records)
        x_ns.append(n_value)
        y_rates.append(overall_rate)
        # mean delay across available records
        valid_delays = [
            r.dispatch_timestamp - r.ready_timestamp
            for r in records
            if r.ready_timestamp is not None
        ]
        mean_delay = mean(valid_delays) if valid_delays else 0.0
        y_mean_delays.append(mean_delay)

    # Plot line charts using plotext if available; fallback prints otherwise
    try:
        import plotext as plt  # type: ignore

        # Session dispatch rate vs n
        plt.clear_figure()
        plt.plotsize(100, 30)
        plt.title("Session dispatch rate vs n")
        plt.xlabel("n (first sessions considered)")
        plt.ylabel("rate (/s)")
        plt.plot(x_ns, y_rates, marker="dot")
        plt.show()
        
        print("--------------------------------")

        # Mean ready->dispatch delay vs n
        plt.clear_figure()
        plt.plotsize(100, 30)
        plt.title("Mean ready→dispatch delay vs n (s)")
        plt.xlabel("n (first sessions considered)")
        plt.ylabel("delay (s)")
        plt.plot(x_ns, y_mean_delays, marker="dot")
        plt.show()
    except Exception:
        # Fallback: print compact table
        print("n, rate_per_s, mean_delay_s")
        for n_val, r_val, d_val in zip(x_ns, y_rates, y_mean_delays):
            print(f"{n_val}, {r_val:.6f}, {d_val:.6f}")


if __name__ == "__main__":
    main()


