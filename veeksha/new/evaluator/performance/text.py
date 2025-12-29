import json
import os
import threading
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from veeksha.logger import init_logger
from veeksha.metrics.cdf_sketch import CDFSketch
from veeksha.metrics.metric_utils import (
    find_min_tbc_deadline_to_meet,
    get_deadline_miss_rate_for_target_tbc_values,
    get_request_level_deadline_miss_rate,
    get_throughput_metrics,
)
from veeksha.new.config.evaluator import (
    PerformanceEvaluatorConfig,
    TextChannelPerformanceConfig,
)
from veeksha.new.evaluator.base import EvaluationResult
from veeksha.new.types import ChannelModality

logger = init_logger(__name__)

TARGET_TBC_RANGE = [i * 0.001 for i in range(1, 101)]
QUANTILE_FOR_DEADLINE_MISS_RATE = 0.99


@dataclass
class TextRequestMetrics:
    """Metrics for a single text request."""

    request_id: int
    session_id: int
    request_dispatched_at: float
    completed_at: float  # Actual completion timestamp
    num_prompt_tokens: int
    num_output_tokens: int
    inter_chunk_times: List[float]
    num_requested_output_tokens: Optional[int] = None
    session_total_requests: Optional[int] = None
    num_delta_prompt_tokens: Optional[int] = None
    num_total_prompt_tokens: Optional[int] = None
    target_num_delta_prompt_tokens: Optional[int] = None

    @property
    def num_total_tokens(self) -> int:
        if self.num_total_prompt_tokens is not None:
            return self.num_total_prompt_tokens + self.num_output_tokens
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def end_to_end_latency(self) -> float:
        return sum(self.inter_chunk_times)

    @property
    def normalized_end_to_end_latency(self) -> float:
        if self.num_output_tokens == 0:
            return 0.0
        return self.end_to_end_latency / self.num_output_tokens

    @property
    def ttfc(self) -> float:
        if not self.inter_chunk_times:
            return 0.0
        return self.inter_chunk_times[0]

    @property
    def tpot(self) -> float:
        if self.num_output_tokens <= 1:
            return 0.0
        # (E2E - TTFC) / (OutputTokens - 1)
        return (self.end_to_end_latency - self.ttfc) / (self.num_output_tokens - 1)

    @property
    def tbc(self) -> float:
        if len(self.inter_chunk_times) < 2:
            return 0.0
        return sum(self.inter_chunk_times[1:]) / len(self.inter_chunk_times[1:])

    @property
    def output_throughput(self) -> float:
        if self.end_to_end_latency == 0:
            return 0.0
        return self.num_output_tokens / self.end_to_end_latency


class TextPerformanceEvaluator:
    """Performance evaluator for text generation (implements legacy MetricStore)

    - CDFSketch-based metric aggregation
    - Request-level metrics tracking
    - Session metrics (size, duration, dispatch gap, think time)
    - Deadline miss rate calculations
    - Throughput metrics
    - Output storage (CSV, JSON, plots)
    - WandB integration
    - Streaming metrics support
    """

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        channel_config: Optional[TextChannelPerformanceConfig] = None,
        benchmark_start_time: float = 0.0,
    ):
        self.config = config
        self.channel_config = channel_config or TextChannelPerformanceConfig()
        self.benchmark_start_time = benchmark_start_time

        self.lock = threading.Lock()

        # deadlines
        self.ttfc_deadline = self.channel_config.ttfc_deadline
        self.tbc_deadline = self.channel_config.tbc_deadline
        self.target_deadline_miss_rate = self.channel_config.target_deadline_miss_rate

        self.service_level_missed_deadlines: int = 0
        self.service_level_total_deadlines: int = 0

        # request tracking
        self._pending_requests: Dict[int, Dict[str, Any]] = (
            {}
        )  # request_id -> dispatch info

        # aggregate metrics
        self.summaries: Dict[str, CDFSketch] = {
            "num_prompt_tokens": CDFSketch(
                metric_name="Number of Prompt Tokens",
                should_write_to_wandb=config.wandb_enabled,
            ),
            "num_output_tokens": CDFSketch(
                metric_name="Number of Output Tokens",
                should_write_to_wandb=config.wandb_enabled,
            ),
            "num_total_tokens": CDFSketch(
                metric_name="Number of Total Tokens",
                should_write_to_wandb=config.wandb_enabled,
            ),
            "tpot": CDFSketch(
                metric_name="Time per Output Token",
                should_write_to_wandb=config.wandb_enabled,
                unit="s",
            ),
            "ttfc": CDFSketch(
                metric_name="Time to First Chunk",
                should_write_to_wandb=config.wandb_enabled,
                unit="s",
            ),
            "tbc": CDFSketch(
                metric_name="Time Between Chunks",
                should_write_to_wandb=config.wandb_enabled,
                unit="s",
            ),
            "end_to_end_latency": CDFSketch(
                metric_name="End to End Latency",
                should_write_to_wandb=config.wandb_enabled,
                unit="s",
            ),
            "normalized_end_to_end_latency": CDFSketch(
                metric_name="Normalized End to End Latency",
                should_write_to_wandb=config.wandb_enabled,
                unit="s/token",
            ),
            "output_throughput": CDFSketch(
                metric_name="Output Throughput",
                should_write_to_wandb=config.wandb_enabled,
            ),
            "deadline_miss_rate": CDFSketch(
                metric_name=f"Deadline Miss Rate ({self.tbc_deadline}s TBC, {self.ttfc_deadline}s TTFC)",
                should_write_to_wandb=config.wandb_enabled,
            ),
            "min_tbc_deadline_to_meet": CDFSketch(
                metric_name=f"Min TBC Deadline to Meet {self.target_deadline_miss_rate * 100}% Miss Rate",
                should_write_to_wandb=config.wandb_enabled,
            ),
            "session_size": CDFSketch(
                metric_name="Requests per Session",
                should_write_to_wandb=config.wandb_enabled,
            ),
            "session_duration": CDFSketch(
                metric_name="Session Duration",
                should_write_to_wandb=config.wandb_enabled,
                unit="s",
            ),
            "session_dispatch_gap": CDFSketch(
                metric_name="Gap Between Session Starts",
                should_write_to_wandb=config.wandb_enabled,
                unit="s",
            ),
            "session_think_time": CDFSketch(
                metric_name="Intra-session Think Time",
                should_write_to_wandb=config.wandb_enabled,
                unit="s",
            ),
        }

        # request-level metrics
        self._request_level_summary_keys = {
            "num_prompt_tokens",
            "num_output_tokens",
            "num_total_tokens",
            "tpot",
            "ttfc",
            "tbc",
            "end_to_end_latency",
            "normalized_end_to_end_latency",
            "output_throughput",
            "deadline_miss_rate",
            "min_tbc_deadline_to_meet",
        }

        self.request_dispatched_at: List[float] = []
        self.completed_at: List[float] = []
        self.num_prompt_tokens: List[int] = []
        self.num_output_tokens: List[int] = []
        self.num_requested_output_tokens: List[Optional[int]] = []
        self.num_delta_prompt_tokens: List[Optional[int]] = []
        self.num_total_prompt_tokens: List[Optional[int]] = []
        self.target_num_delta_prompt_tokens: List[Optional[int]] = []
        self.num_total_tokens: List[int] = []
        self.tpot: List[float] = []
        self.ttfc: List[float] = []
        self.tbc: List[List[float]] = []
        self.end_to_end_latency: List[float] = []
        self.normalized_end_to_end_latency: List[float] = []
        self.output_throughput: List[float] = []
        self.deadline_miss_rate: List[float] = []
        self.min_tbc_deadline_to_meet: List[float] = []
        self.session_ids: List[Optional[int]] = []
        self.session_total_requests: List[Optional[int]] = []
        self.request_ids: List[int] = []

        # streaming
        self._request_rows_streamed: int = 0
        self._request_time_reference: float = self.benchmark_start_time

        # session tracking
        self._session_last_completion: Dict[int, float] = {}

    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        content: Any,
    ) -> None:
        """Register a text request that was dispatched."""
        with self.lock:
            if self._request_time_reference == 0.0:
                self._request_time_reference = dispatched_at

            target_output_tokens = getattr(content, "target_output_tokens", None)
            target_prompt_tokens = getattr(content, "target_prompt_tokens", None)

            self._pending_requests[request_id] = {
                "session_id": session_id,
                "dispatched_at": dispatched_at,
                "target_output_tokens": target_output_tokens,
                "target_prompt_tokens": target_prompt_tokens,
            }

    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
    ) -> None:
        """Record that a text request completed."""
        with self.lock:
            # get dispatch info
            dispatch_info = self._pending_requests.pop(request_id, None)
            if dispatch_info is None:
                logger.warning(f"Request {request_id} completed but was not registered")
                return

            dispatched_at = dispatch_info["dispatched_at"]
            target_output_tokens = dispatch_info.get("target_output_tokens")
            target_prompt_tokens = dispatch_info.get("target_prompt_tokens")

            # Extract metrics from the text channel response
            channel_response = response.channels.get(ChannelModality.TEXT)

            if channel_response is not None:
                channel_metrics = channel_response.metrics or {}
                num_total_prompt_tokens = channel_metrics.get("num_total_prompt_tokens")
                num_delta_prompt_tokens = channel_metrics.get("num_delta_prompt_tokens")
                num_prompt_tokens = num_delta_prompt_tokens or 0
                num_output_tokens = channel_metrics.get("num_output_tokens", 0)
                inter_chunk_times = channel_metrics.get("inter_chunk_times", [])
            else:
                num_prompt_tokens = 0
                num_output_tokens = 0
                inter_chunk_times = []
                num_delta_prompt_tokens = None
                num_total_prompt_tokens = None

            session_total_requests = getattr(response, "session_total_requests", None)

            # Create metrics object
            metrics = TextRequestMetrics(
                request_id=request_id,
                session_id=session_id,
                request_dispatched_at=dispatched_at,
                completed_at=completed_at,
                num_prompt_tokens=num_prompt_tokens,
                num_output_tokens=num_output_tokens,
                inter_chunk_times=inter_chunk_times,
                num_requested_output_tokens=target_output_tokens,
                session_total_requests=session_total_requests,
                num_delta_prompt_tokens=num_delta_prompt_tokens,
                num_total_prompt_tokens=num_total_prompt_tokens,
                target_num_delta_prompt_tokens=target_prompt_tokens,
            )

            prev_completion = self._session_last_completion.get(session_id)
            if prev_completion is not None:
                think_time = dispatched_at - prev_completion
                if think_time >= 0:
                    self.summaries["session_think_time"].put(think_time)
            self._session_last_completion[session_id] = completed_at

            # Update CDF sketches
            self._update_summaries(metrics)

            # Store request-level metrics
            self._store_request_metrics(metrics, dispatched_at)

    def _update_summaries(self, metrics: TextRequestMetrics) -> None:
        """Update CDF sketches with request metrics."""
        for metric_name, cdf_sketch in self.summaries.items():
            if metric_name not in self._request_level_summary_keys:
                continue

            if metric_name == "tbc":
                # TBC is the inter-chunk times excluding TTFC
                cdf_sketch.extend(metrics.inter_chunk_times[1:])
            elif metric_name == "deadline_miss_rate":
                (
                    deadline_miss_rate,
                    missed_deadlines,
                    total_deadlines,
                ) = get_request_level_deadline_miss_rate(
                    inter_chunk_times=metrics.inter_chunk_times,
                    ttfc_deadline=self.ttfc_deadline,
                    tbc_deadline=self.tbc_deadline,
                )
                cdf_sketch.put(deadline_miss_rate)
                self.service_level_missed_deadlines += missed_deadlines
                self.service_level_total_deadlines += total_deadlines
            elif metric_name == "min_tbc_deadline_to_meet":
                cdf_sketch.put(
                    find_min_tbc_deadline_to_meet(
                        inter_chunk_times=metrics.inter_chunk_times,
                        ttfc_deadline=self.ttfc_deadline,
                        target_deadline_miss_rate=self.target_deadline_miss_rate,
                    )
                )
            else:
                cdf_sketch.put(getattr(metrics, metric_name))

    def _store_request_metrics(
        self, metrics: TextRequestMetrics, dispatched_at: float
    ) -> None:
        """Store request-level metrics for detailed output."""
        normalized_dispatched_at = max(
            0.0, dispatched_at - self._request_time_reference
        )

        # Calculate deadline miss rate for this request
        dmr, _, _ = get_request_level_deadline_miss_rate(
            inter_chunk_times=metrics.inter_chunk_times,
            ttfc_deadline=self.ttfc_deadline,
            tbc_deadline=self.tbc_deadline,
        )
        min_tbc = find_min_tbc_deadline_to_meet(
            inter_chunk_times=metrics.inter_chunk_times,
            ttfc_deadline=self.ttfc_deadline,
            target_deadline_miss_rate=self.target_deadline_miss_rate,
        )

        self.request_dispatched_at.append(normalized_dispatched_at)
        self.completed_at.append(
            max(0.0, metrics.completed_at - self._request_time_reference)
        )
        self.num_prompt_tokens.append(metrics.num_prompt_tokens)
        self.num_output_tokens.append(metrics.num_output_tokens)
        self.num_requested_output_tokens.append(metrics.num_requested_output_tokens)
        self.num_delta_prompt_tokens.append(metrics.num_delta_prompt_tokens)
        self.num_total_prompt_tokens.append(metrics.num_total_prompt_tokens)
        self.target_num_delta_prompt_tokens.append(
            metrics.target_num_delta_prompt_tokens
        )
        self.num_total_tokens.append(metrics.num_total_tokens)
        self.tpot.append(metrics.tpot)
        self.ttfc.append(metrics.ttfc)
        self.tbc.append(metrics.inter_chunk_times[1:])
        self.end_to_end_latency.append(metrics.end_to_end_latency)
        self.normalized_end_to_end_latency.append(metrics.normalized_end_to_end_latency)
        self.output_throughput.append(metrics.output_throughput)
        self.deadline_miss_rate.append(dmr)
        self.min_tbc_deadline_to_meet.append(min_tbc)
        self.session_ids.append(metrics.session_id)
        self.session_total_requests.append(metrics.session_total_requests)
        self.request_ids.append(metrics.request_id)

    def record_session_completed(
        self,
        session_id: int,
        session_size: int,
        first_dispatch_at: Optional[float],
        last_completion_at: Optional[float],
    ) -> None:
        """Record session-level metrics."""
        with self.lock:
            # Session size
            self.summaries["session_size"].put(session_size)

            # Session duration
            if first_dispatch_at is not None and last_completion_at is not None:
                duration = max(0.0, last_completion_at - first_dispatch_at)
                self.summaries["session_duration"].put(duration)

            # Clean up session state
            self._session_last_completion.pop(session_id, None)

    def record_session_dispatch_gap(self, gap: float) -> None:
        """Record gap between session starts."""
        with self.lock:
            self.summaries["session_dispatch_gap"].put(gap)

    def get_summary(self) -> Dict[str, float]:
        """Get summary metrics from all CDF sketches."""
        perf_summary = {}
        for cdf_sketch in self.summaries.values():
            perf_summary.update(cdf_sketch.get_summary())

        # Add service-level deadline miss rate
        if self.service_level_total_deadlines > 0:
            service_level_dmr = (
                self.service_level_missed_deadlines / self.service_level_total_deadlines
            )
        else:
            service_level_dmr = 0.0

        perf_summary["Service Level Deadline Miss Rate"] = service_level_dmr
        perf_summary["Service Level Missed Deadlines"] = (
            self.service_level_missed_deadlines
        )
        perf_summary["Service Level Total Deadlines"] = (
            self.service_level_total_deadlines
        )

        return perf_summary

    def finalize(self) -> EvaluationResult:
        """Finalize evaluation and return results."""
        with self.lock:
            return EvaluationResult(
                evaluator_type="text_performance",
                channel=ChannelModality.TEXT,
                metrics=self.get_summary(),
            )

    def get_streaming_metrics(self) -> Optional[Dict[str, Any]]:
        """Return current metrics for streaming."""
        with self.lock:
            return self.get_summary()

    def save(self, output_dir: str) -> None:
        """Save all evaluation artifacts."""
        with self.lock:
            self._save_request_level_metrics(output_dir)
            self._save_cdf_csvs(output_dir)
            self._save_performance_csv(output_dir)
            self._save_throughput_metrics(output_dir)
            self._save_deadline_miss_rate_for_target_tbc(output_dir)
            self._plot_cdfs(output_dir)
            self._store_ttfc_violin_plots(output_dir)
            self._store_generation_stalls(output_dir)

            if self.config.wandb_enabled:
                self._log_wandb_metrics(output_dir)

    def flush_streaming_outputs(self, output_dir: str) -> None:
        """Flush current metrics for streaming."""
        with self.lock:
            # Export new request-level rows
            rows = self._export_request_rows(self._request_rows_streamed)
            if rows:
                self._append_request_level_rows(output_dir, rows)
                self._request_rows_streamed = len(self.ttfc)

            # Save current CDF summaries
            self._save_performance_csv(output_dir)
            self._save_cdf_csvs(output_dir)

    # -------------------------------------------------------------------------
    # Output methods
    # -------------------------------------------------------------------------

    def _save_request_level_metrics(self, output_dir: str) -> None:
        """Save request-level metrics as JSONL."""
        path = os.path.join(output_dir, "request_level_metrics.jsonl")
        rows = self._export_request_rows(0)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row))
                f.write("\n")

    def _export_request_rows(self, start_index: int = 0) -> List[Dict[str, Any]]:
        """Export request-level metrics as list of dicts."""
        rows: List[Dict[str, Any]] = []
        for idx in range(start_index, len(self.ttfc)):
            rows.append(
                {
                    "request_id": self.request_ids[idx],
                    "session_id": self.session_ids[idx],
                    "session_total_requests": self.session_total_requests[idx],
                    "dispatched_at": round(self.request_dispatched_at[idx], 5),
                    "completed_at": round(self.completed_at[idx], 5),
                    "num_delta_prompt_tokens": self.num_delta_prompt_tokens[idx],
                    "num_total_prompt_tokens": self.num_total_prompt_tokens[idx],
                    "target_num_delta_prompt_tokens": self.target_num_delta_prompt_tokens[
                        idx
                    ],
                    "num_output_tokens": self.num_output_tokens[idx],
                    "num_requested_output_tokens": self.num_requested_output_tokens[
                        idx
                    ],
                    "num_total_tokens": self.num_total_tokens[idx],
                    "tpot": round(self.tpot[idx], 5),
                    "ttfc": round(self.ttfc[idx], 5),
                    "end_to_end_latency": round(self.end_to_end_latency[idx], 5),
                    "normalized_end_to_end_latency": round(
                        self.normalized_end_to_end_latency[idx], 5
                    ),
                    "output_throughput": round(self.output_throughput[idx], 5),
                    "deadline_miss_rate": self.deadline_miss_rate[idx],
                    "min_tbc_deadline_to_meet": round(
                        self.min_tbc_deadline_to_meet[idx], 5
                    ),
                    "tbc": [round(t, 5) for t in self.tbc[idx]],
                }
            )
        return rows

    def _append_request_level_rows(
        self, output_dir: str, rows: List[Dict[str, Any]]
    ) -> None:
        """Append request-level rows to JSONL file."""
        path = os.path.join(output_dir, "request_level_metrics.jsonl")
        with open(path, "a") as f:
            for row in rows:
                f.write(json.dumps(row))
                f.write("\n")

    def _save_cdf_csvs(self, output_dir: str) -> None:
        """Save CDF data as CSV files."""
        for metric_name, cdf_sketch in self.summaries.items():
            df = cdf_sketch._to_df()
            df.to_csv(os.path.join(output_dir, f"{metric_name}.csv"), index=False)

    def _save_performance_csv(self, output_dir: str) -> None:
        """Save performance summary as CSV."""
        path = os.path.join(output_dir, "perf_metrics.csv")
        header = self.summaries["num_prompt_tokens"].get_csv_header()
        rows = [header]
        for cdf_sketch in self.summaries.values():
            rows.append(cdf_sketch.to_csv_row())
        with open(path, "w") as f:
            f.write("\n".join(rows))

    def _save_throughput_metrics(self, output_dir: str) -> None:
        """Save throughput metrics."""
        tpot_based, tbc_based, deadline_based = get_throughput_metrics(
            self.tpot, self.tbc
        )
        metrics = {
            "tpot_based_throughput": tpot_based,
            "tbc_based_throughput": tbc_based,
            "deadline_based_throughput": deadline_based,
        }
        path = os.path.join(output_dir, "throughput_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    def _save_deadline_miss_rate_for_target_tbc(self, output_dir: str) -> None:
        """Save deadline miss rate for various TBC targets."""
        deadline_miss_rates = get_deadline_miss_rate_for_target_tbc_values(
            tbc_times=self.tbc,
            target_tbc_deadline_array=TARGET_TBC_RANGE,
            quantile=QUANTILE_FOR_DEADLINE_MISS_RATE,
        )

        percentile_value = int(QUANTILE_FOR_DEADLINE_MISS_RATE * 100)
        data = {
            "Target TBC (ms)": [int(i * 1000) for i in TARGET_TBC_RANGE],
            f"Miss Rate P({percentile_value})": deadline_miss_rates,
        }

        path = os.path.join(
            output_dir,
            f"p{percentile_value}_deadline_miss_rate_for_target_tbc_values.json",
        )
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _plot_cdfs(self, output_dir: str) -> None:
        """Generate CDF plots for all metrics."""
        for metric_name, cdf_sketch in self.summaries.items():
            cdf_sketch.plot_cdf(output_dir, metric_name)

    def _store_ttfc_violin_plots(self, output_dir: str) -> None:
        """Save TTFC distribution vs prompt length plots."""
        if not self.ttfc:
            return

        try:
            import rekha as rk

            from veeksha.metrics.plot_utils import (
                apply_axis_scale,
                format_axis_label,
                recommend_axis_scale,
            )

            prompt_lengths = [int(n) for n in self.num_prompt_tokens]
            ttfcs = list(self.ttfc)
            if len(prompt_lengths) != len(ttfcs):
                return

            base_df = pd.DataFrame({"prompt_length": prompt_lengths, "ttfc": ttfcs})

            min_len = int(base_df["prompt_length"].min())
            max_len = int(base_df["prompt_length"].max())

            if max_len <= min_len:
                base_df["prompt_length_bin"] = pd.Series(
                    [f"{min_len}-{max_len}"] * len(base_df)
                ).astype("category")
            else:
                target_bins = 12
                bins = max(5, min(20, target_bins))
                raw_edges = np.linspace(min_len, max_len, bins + 1)
                int_edges = np.unique(np.round(raw_edges).astype(int))

                if int_edges.size < 2:
                    base_df["prompt_length_bin"] = pd.Series(
                        [f"{min_len}-{max_len}"] * len(base_df)
                    ).astype("category")
                else:
                    edges = int_edges
                    bins = edges.size - 1
                    labels = [f"{edges[i]}-{edges[i + 1]}" for i in range(bins)]
                    base_df["prompt_length_bin"] = pd.cut(
                        base_df["prompt_length"],
                        bins=edges,
                        include_lowest=True,
                        labels=labels,
                        right=True,
                    )

            df = base_df[["prompt_length_bin", "ttfc"]].copy()
            df["prompt_length_bin"] = df["prompt_length_bin"].astype("category")

            ttfc_scale = recommend_axis_scale(df["ttfc"], kind="numeric")
            y_label = "TTFC (s)"

            fig = rk.box(
                df,
                x="prompt_length_bin",
                y="ttfc",
                labels={
                    "prompt_length_bin": "Number of Prompt Tokens",
                    "ttfc": y_label,
                },
            )
            fig.save(os.path.join(output_dir, "ttfc_violin_plot.png"))

            if ttfc_scale != "linear":
                fig_scaled = rk.box(
                    df,
                    x="prompt_length_bin",
                    y="ttfc",
                    labels={
                        "prompt_length_bin": "Number of Prompt Tokens",
                        "ttfc": format_axis_label("TTFC", "s", ttfc_scale),
                    },
                )
                apply_axis_scale(fig_scaled, axis="y", scale=ttfc_scale)
                suffix = "log" if ttfc_scale == "log" else "symlog"
                fig_scaled.save(
                    os.path.join(output_dir, f"ttfc_violin_plot_{suffix}_y.png")
                )

        except Exception as e:
            logger.warning(f"Failed to generate TTFC violin plots: {e}")

    def _store_generation_stalls(self, output_dir: str, request_idx: int = 0) -> None:
        """Save tokens generated vs time plot for a sample request."""
        if request_idx >= len(self.ttfc):
            return

        try:
            import rekha as rk

            from veeksha.metrics.plot_utils import (
                apply_axis_scale,
                format_axis_label,
                recommend_axis_scale,
            )

            token_times = [self.ttfc[request_idx]] + self.tbc[request_idx]
            token_times_cumulative = list(accumulate(token_times))
            tokens_generated = list(range(1, len(token_times_cumulative) + 1))

            time_scale = recommend_axis_scale(token_times_cumulative, kind="numeric")
            x_label = "Time (s)"

            df = pd.DataFrame(
                {
                    x_label: token_times_cumulative,
                    "Tokens Generated": tokens_generated,
                }
            )

            fig = rk.line(
                df,
                x=x_label,
                y="Tokens Generated",
                title="Tokens Generated vs Time",
            )
            fig.save(os.path.join(output_dir, "tokens_generated_vs_time.png"))

            if time_scale != "linear":
                scaled_label = format_axis_label("Time", "s", time_scale)
                df_scaled = pd.DataFrame(
                    {
                        scaled_label: token_times_cumulative,
                        "Tokens Generated": tokens_generated,
                    }
                )
                fig_scaled = rk.line(
                    df_scaled,
                    x=scaled_label,
                    y="Tokens Generated",
                    title="Tokens Generated vs Time",
                )
                apply_axis_scale(fig_scaled, axis="x", scale=time_scale)
                suffix = "log" if time_scale == "log" else "symlog"
                fig_scaled.save(
                    os.path.join(output_dir, f"tokens_generated_vs_time_{suffix}_x.png")
                )

        except Exception as e:
            logger.warning(f"Failed to generate stall plots: {e}")

    def _log_wandb_metrics(self, output_dir: str) -> None:
        """Log metrics to Weights & Biases."""
        try:
            import wandb

            if not wandb.run:
                return

            # Log summary table
            summary_path = os.path.join(output_dir, "summary_stats.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                numeric_rows = [
                    {"Metric": k, "Value": float(v)}
                    for k, v in summary.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ]
                if numeric_rows:
                    df = pd.DataFrame.from_records(numeric_rows)
                    wandb.log({"summary_stats_table": wandb.Table(dataframe=df)})

            # Log throughput metrics
            throughput_path = os.path.join(output_dir, "throughput_metrics.json")
            if os.path.exists(throughput_path):
                with open(throughput_path, "r") as f:
                    throughput = json.load(f)
                data = {
                    "Metric Type": ["TPOT Based", "TBC Based", "Deadline Based"],
                    "Throughput (tok/s)": [
                        throughput.get("tpot_based_throughput", 0),
                        throughput.get("tbc_based_throughput", 0),
                        throughput.get("deadline_based_throughput", 0),
                    ],
                }
                df = pd.DataFrame(data)
                wandb.log(
                    {
                        "throughput_metrics": wandb.plot.bar(
                            table=wandb.Table(dataframe=df),
                            label="Metric Type",
                            value="Throughput (tok/s)",
                            title="Token Throughput",
                        )
                    }
                )

            # Log TTFC/TBC scalar charts
            self._log_ttfc_tbc_scalar_charts()

            # Log images
            for plot_name in ["ttfc_violin_plot.png", "tokens_generated_vs_time.png"]:
                plot_path = os.path.join(output_dir, plot_name)
                if os.path.exists(plot_path):
                    wandb.log({plot_name.replace(".png", ""): wandb.Image(plot_path)})

        except Exception as e:
            logger.warning(f"Failed to log WandB metrics: {e}")

    def _log_ttfc_tbc_scalar_charts(self) -> None:
        """Log TTFC and TBC scalar charts to WandB."""
        try:
            import wandb

            if not wandb.run:
                return

            def log_for_sketch(sketch_key: str, short_name: str) -> None:
                if sketch_key not in self.summaries:
                    return
                sketch = self.summaries[sketch_key].sketch
                if sketch.count == 0:
                    return
                try:
                    stats = {
                        "Min": sketch._min,
                        "Mean": sketch.avg,
                        "Median": sketch.get_quantile_value(0.5),
                        "P90": sketch.get_quantile_value(0.9),
                        "P99": sketch.get_quantile_value(0.99),
                        "Max": sketch._max,
                    }
                    for stat_name, stat_value in stats.items():
                        df = pd.DataFrame(
                            {"Label": [short_name], "Value": [float(stat_value)]}
                        )
                        wandb.log(
                            {
                                f"{short_name} {stat_name}": wandb.plot.bar(
                                    table=wandb.Table(dataframe=df),
                                    label="Label",
                                    value="Value",
                                    title=f"{short_name} {stat_name}",
                                )
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to compute stats for {short_name}: {e}")

            log_for_sketch("ttfc", "TTFC (s)")
            log_for_sketch("tbc", "TBC (s)")

        except Exception as e:
            logger.warning(f"Failed to log scalar charts: {e}")
