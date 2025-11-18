import json
import os
import threading
from collections import defaultdict
from itertools import accumulate
from typing import Any, DefaultDict, Dict, List, Optional

import numpy as np
import pandas as pd
import rekha as rk
import wandb

from veeksha.config.metrics import MetricsConfig
from veeksha.logger import init_logger
from veeksha.metrics.cdf_sketch import CDFSketch
from veeksha.metrics.metric_utils import (
    find_min_tbt_deadline_to_meet,
    get_deadline_miss_rate_for_target_tbt_values,
    get_request_level_deadline_miss_rate,
    get_throughput_metrics,
)
from veeksha.metrics.plot_utils import (
    apply_axis_scale,
    format_axis_label,
    recommend_axis_scale,
)
from veeksha.metrics.request_level_metrics import RequestLevelMetrics
from veeksha.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)


TARGET_TBT_RANGE = [i * 0.001 for i in range(1, 101)]
QUANTILE_FOR_DEADLINE_MISS_RATE = 0.99


class MetricStore:
    def __init__(
        self,
        timeout: float,
        max_requests: int,
        metrics_config: MetricsConfig,
    ) -> None:
        self.timeout = timeout
        self.max_requests = max_requests

        self.lock = threading.Lock()
        self.num_requests: int = 0
        self.num_errored_requests: int = 0
        self.num_completed_requests: int = 0
        self.num_cancelled_requests: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.error_code_freq: DefaultDict[int, int] = defaultdict(int)
        self.ttft_deadline: float = metrics_config.deadline_report.ttft_deadline
        self.tbt_deadline: float = metrics_config.deadline_report.tbt_deadline
        self.target_deadline_miss_rate: float = (
            metrics_config.deadline_report.target_deadline_miss_rate
        )
        self.service_level_missed_deadlines: int = 0
        self.service_level_total_deadlines: int = 0
        self.should_write_metrics_to_wandb: bool = (
            metrics_config.should_write_metrics_to_wandb
        )
        self.wandb_project: Optional[str] = metrics_config.wandb_project
        self.wandb_group: Optional[str] = metrics_config.wandb_group
        self.wandb_run_name: Optional[str] = metrics_config.wandb_run_name
        self.output_dir: Optional[str] = metrics_config.output_dir
        self.stream_metrics_enabled: bool = metrics_config.stream_metrics
        self.stream_metrics_interval: float = metrics_config.stream_metrics_interval

        self._stream_trigger = threading.Event()
        self._stream_stop_event = threading.Event()
        self._stream_has_updates = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        self._request_rows_streamed: int = 0
        self._request_level_stream_path: Optional[str] = None

        self.request_level_metrics = RequestLevelMetrics(
            deadline_config=metrics_config.deadline_report,
        )

        self.summaries: Dict[str, CDFSketch] = {
            "num_prompt_tokens": CDFSketch(
                metric_name="Number of Prompt Tokens",
                should_write_to_wandb=self.should_write_metrics_to_wandb,
            ),
            "num_output_tokens": CDFSketch(
                "Number of Output Tokens", self.should_write_metrics_to_wandb
            ),
            "num_total_tokens": CDFSketch(
                "Number of Total Tokens", self.should_write_metrics_to_wandb
            ),
            "tpot": CDFSketch(
                metric_name="Time per Output Token",
                should_write_to_wandb=self.should_write_metrics_to_wandb,
                unit="s",
            ),
            "ttft": CDFSketch(
                metric_name="Time to First Token",
                should_write_to_wandb=self.should_write_metrics_to_wandb,
                unit="s",
            ),
            "tbt": CDFSketch(
                metric_name="Time Between Tokens",
                should_write_to_wandb=self.should_write_metrics_to_wandb,
                unit="s",
            ),
            "end_to_end_latency": CDFSketch(
                metric_name="End to End Latency",
                should_write_to_wandb=self.should_write_metrics_to_wandb,
                unit="s",
            ),
            "normalized_end_to_end_latency": CDFSketch(
                metric_name="Normalized End to End Latency",
                should_write_to_wandb=self.should_write_metrics_to_wandb,
                unit="s/token",
            ),
            "output_throughput": CDFSketch(
                metric_name="Output Throughput",
                should_write_to_wandb=self.should_write_metrics_to_wandb,
            ),
            "deadline_miss_rate": CDFSketch(
                f"Deadline Miss Rate with {self.tbt_deadline}s TBT Deadline, {self.ttft_deadline}s TTFT Deadline",
                self.should_write_metrics_to_wandb,
            ),
            "min_tbt_deadline_to_meet": CDFSketch(
                f"Min Deadline to Meet Target Deadline Miss Rate of {self.target_deadline_miss_rate * 100}%",
                self.should_write_metrics_to_wandb,
            ),
        }

        self._init_wandb()
        if self.stream_metrics_enabled:
            self._start_metric_streamer()

    def _init_wandb(self):
        if not self.should_write_metrics_to_wandb:
            logger.info("wandb disabled; not initialized")
            return

        # Prepend group to run name if group is specified
        run_name = self.wandb_run_name
        if self.wandb_group and self.wandb_run_name:
            run_name = f"{self.wandb_group}_{self.wandb_run_name}"

        wandb.init(
            project=self.wandb_project,
            group=self.wandb_group,
            name=run_name,
            config={
                "timeout": self.timeout,
                "max_requests": self.max_requests,
                "ttft_deadline": self.ttft_deadline,
                "tbt_deadline": self.tbt_deadline,
                "target_deadline_miss_rate": self.target_deadline_miss_rate,
            },
        )
        logger.info("wandb enabled")

    def _start_metric_streamer(self) -> None:
        if not self.output_dir:
            logger.warning(
                "stream_metrics enabled but output directory is missing; disabling streaming."
            )
            self.stream_metrics_enabled = False
            return

        os.makedirs(self.output_dir, exist_ok=True)
        self._request_level_stream_path = os.path.join(
            self.output_dir, "request_level_metrics.jsonl"
        )
        with open(self._request_level_stream_path, "w") as stream_file:
            stream_file.write("")

        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="metric-store-streamer",
            daemon=True,
        )
        self._stream_thread.start()
        logger.info(
            "Metric streaming enabled; flushing artifacts every %.1fs",
            self.stream_metrics_interval,
        )

    def _stream_loop(self) -> None:
        while True:
            triggered = self._stream_trigger.wait(timeout=self.stream_metrics_interval)
            if triggered:
                self._stream_trigger.clear()
            if triggered or self._stream_has_updates.is_set():
                try:
                    self._flush_streaming_outputs()
                    self._stream_has_updates.clear()
                except Exception as exc:
                    logger.warning(f"Streaming metrics flush failed: {exc}")
            if self._stream_stop_event.is_set():
                # ensure final flush on shutdown
                try:
                    self._flush_streaming_outputs()
                    self._stream_has_updates.clear()
                except Exception as exc:
                    logger.warning(f"Final streaming flush failed: {exc}")
                break

    def _flush_streaming_outputs(self) -> None:
        if not (self.stream_metrics_enabled and self.output_dir):
            return

        with self.lock:
            perf_header = self.summaries["num_prompt_tokens"].get_csv_header()
            perf_rows = [perf_header]
            for cdf_sketch in self.summaries.values():
                perf_rows.append(cdf_sketch.to_csv_row())

            summary_stats = {
                **self.get_summary(),
                "error_code_freq": dict(self.error_code_freq),
            }

            if self.service_level_total_deadlines > 0:
                service_level_deadline_miss_rate = (
                    self.service_level_missed_deadlines
                    / self.service_level_total_deadlines
                )
            else:
                service_level_deadline_miss_rate = 0.0

            service_level_metrics = {
                "service_level_missed_deadlines": self.service_level_missed_deadlines,
                "service_level_total_deadlines": self.service_level_total_deadlines,
                "service_level_deadline_miss_rate": service_level_deadline_miss_rate,
            }

            cdf_frames = {
                metric_name: metric_summary._to_df()
                for metric_name, metric_summary in self.summaries.items()
            }

            request_rows: List[Dict[str, Any]]
            request_rows_end: int
            request_rows, request_rows_end = self.request_level_metrics.export_rows(
                self._request_rows_streamed
            )

        perf_path = os.path.join(self.output_dir, "perf_metrics.csv")
        with open(perf_path, "w") as perf_file:
            perf_file.write("\n".join(perf_rows))

        self._write_summary_stats_stream(summary_stats)
        self._write_service_level_metrics_stream(service_level_metrics)
        self._write_cdf_csvs(cdf_frames)

        if request_rows:
            self._append_request_level_rows(request_rows)
            self._request_rows_streamed = request_rows_end

    def _write_summary_stats_stream(self, summary_stats: Dict[str, float]) -> None:
        if not self.output_dir:
            return
        summary_stats_path = os.path.join(self.output_dir, "summary_stats.json")
        with open(summary_stats_path, "w") as summary_file:
            json.dump(summary_stats, summary_file)

    def _write_service_level_metrics_stream(self, metrics: Dict[str, float]) -> None:
        if not self.output_dir:
            return
        json_path = os.path.join(self.output_dir, "service_level_metrics.json")
        with open(json_path, "w") as json_file:
            json.dump(metrics, json_file)

    def _write_cdf_csvs(self, cdf_frames: Dict[str, pd.DataFrame]) -> None:
        if not self.output_dir:
            return
        for metric_name, df in cdf_frames.items():
            df.to_csv(os.path.join(self.output_dir, f"{metric_name}.csv"), index=False)

    def _append_request_level_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows or not self._request_level_stream_path:
            return
        with open(self._request_level_stream_path, "a") as stream_file:
            for row in rows:
                stream_file.write(json.dumps(row))
                stream_file.write("\n")

    def _shutdown_metric_streamer(self) -> None:
        if not self._stream_thread:
            return
        self._stream_trigger.set()
        self._stream_stop_event.set()
        self._stream_thread.join()
        self._stream_thread = None
        self.stream_metrics_enabled = False

    def _persist_wandb_run_info(self, output_dir: str) -> None:
        """Persist basic wandb run identifiers for downstream consumers.

        This allows external orchestrators (e.g., capacity search) to
        reference the exact wandb run for tagging or dashboards.
        """
        try:
            if not (self.should_write_metrics_to_wandb and wandb.run):
                return

            run_info = {
                "id": getattr(wandb.run, "id", None),
                "name": getattr(wandb.run, "name", None),
                "entity": getattr(wandb.run, "entity", None),
                "project": getattr(wandb.run, "project", None),
                "group": getattr(wandb.run, "group", None),
                "path": getattr(wandb.run, "path", None),
                "url": getattr(wandb.run, "url", None),
            }
            with open(os.path.join(output_dir, "wandb_run.json"), "w") as f:
                json.dump(run_info, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist wandb run info: {e}")

    @property
    def error_rate(self):
        return (
            self.num_errored_requests / self.num_requests
            if self.num_requests > 0
            else 0.0
        )

    def register_launched_request(self):
        with self.lock:
            self.num_requests += 1

    def add_request_metrics(self, request_metrics: RequestMetrics):
        with self.lock:
            if request_metrics.cancelled:
                self.num_cancelled_requests += 1
                return
            if request_metrics.error_code is not None or request_metrics.error_msg:
                if request_metrics.error_code is not None:
                    self.error_code_freq[request_metrics.error_code] += 1
                self.num_errored_requests += 1
                return

            self.num_completed_requests += 1

            for metric_name, cdf_sketch in self.summaries.items():
                if metric_name == "tbt":
                    cdf_sketch.extend(request_metrics.inter_token_times[1:])
                elif metric_name == "deadline_miss_rate":
                    (
                        deadline_miss_rate,
                        missed_deadlines,
                        total_deadlines,
                    ) = get_request_level_deadline_miss_rate(
                        inter_token_times=request_metrics.inter_token_times,
                        ttft_deadline=self.ttft_deadline,
                        tbt_deadline=self.tbt_deadline,
                    )
                    cdf_sketch.put(deadline_miss_rate)
                    self.service_level_missed_deadlines += missed_deadlines
                    self.service_level_total_deadlines += total_deadlines
                elif metric_name == "min_tbt_deadline_to_meet":
                    cdf_sketch.put(
                        find_min_tbt_deadline_to_meet(
                            inter_token_times=request_metrics.inter_token_times,
                            ttft_deadline=self.ttft_deadline,
                            target_deadline_miss_rate=self.target_deadline_miss_rate,
                        )
                    )
                else:
                    cdf_sketch.put(getattr(request_metrics, metric_name))

            # Record full request-level metrics for successful requests
            self.request_level_metrics.put(request_metrics)

        if self.stream_metrics_enabled and self._stream_thread:
            self._stream_has_updates.set()

    def get_aggregated_summary(self) -> Dict[str, float]:
        return {
            "Number of Requests": self.num_requests,
            "Number of Errored Requests": self.num_errored_requests,
            "Number of Completed Requests": self.num_completed_requests,
            "Number of Cancelled Requests": self.num_cancelled_requests,
            "Error Rate": self.error_rate,
            "Cancellation Rate": (
                self.num_cancelled_requests / self.num_requests
                if self.num_requests > 0
                else 0.0
            ),
            "Deadline Miss Rate": (
                self.service_level_missed_deadlines / self.service_level_total_deadlines
                if self.service_level_total_deadlines > 0
                else 0.0
            ),
        }

    def get_summary(self) -> Dict[str, float]:
        perf_summary = {}

        for cdf_sketch in self.summaries.values():
            perf_summary.update(cdf_sketch.get_summary())

        return {
            **self.get_aggregated_summary(),
            **perf_summary,
        }

    def store_output(self, output_dir: str):
        if self.stream_metrics_enabled:
            self._shutdown_metric_streamer()

        perf_csv_path = os.path.join(output_dir, "perf_metrics.csv")
        summary_stats_path = os.path.join(output_dir, "summary_stats.json")

        # store request level metrics as JSONL (final full dump)
        self.request_level_metrics.save_jsonl(output_dir)

        # store metric objects
        logger.info("Storing metric artifacts.")
        for metric_name, metric_summary in self.summaries.items():
            metric_summary._save_df(metric_summary._to_df(), output_dir, metric_name)
            metric_summary.plot_cdf(output_dir, metric_name)

        # store service level deadline stats
        with open(os.path.join(output_dir, "service_level_metrics.json"), "w") as f:
            json.dump(
                {
                    "service_level_missed_deadlines": self.service_level_missed_deadlines,
                    "service_level_total_deadlines": self.service_level_total_deadlines,
                    "service_level_deadline_miss_rate": (
                        self.service_level_missed_deadlines
                        / self.service_level_total_deadlines
                        if self.service_level_total_deadlines > 0
                        else 0.0
                    ),
                },
                f,
            )

        # store performance metrics
        perf_header = self.summaries["num_prompt_tokens"].get_csv_header()
        perf_rows = [perf_header]
        for cdf_sketch in self.summaries.values():
            perf_rows.append(cdf_sketch.to_csv_row())

        with open(perf_csv_path, "w") as f:
            f.write("\n".join(perf_rows))

        # store summary stats
        with open(summary_stats_path, "w") as f:
            json.dump(
                {**self.get_summary(), "error_code_freq": dict(self.error_code_freq)}, f
            )

        # log summary stats table to wandb
        self.store_summary_stats_table(output_dir)

        # store additional outputs
        self.store_additional_outputs(output_dir)

        # log selected result files as wandb artifacts
        self._log_artifact_files(output_dir)

        # persist run identifiers and finish the run (if enabled)
        if self.should_write_metrics_to_wandb and wandb.run:
            self._persist_wandb_run_info(output_dir)
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"wandb.finish() failed: {e}")

    def store_additional_outputs(self, output_dir: str):
        self.store_deadline_miss_rate_for_target_tbt(output_dir)
        self.store_throughput_metrics(output_dir)
        self.store_ttft_violin_plots(output_dir)
        self.store_generation_stalls(output_dir)
        self.store_ttft_tbt_scalar_charts()

    def _log_artifact_files(self, output_dir: str) -> None:
        if not (self.should_write_metrics_to_wandb and wandb.run):
            return

        artifact = wandb.Artifact(
            name=f"benchmark-output-files-{wandb.run.id}",
            type="benchmark-metrics",
        )

        files_to_log = [
            "config.yml",
            "request_level_metrics.jsonl",
            "service_level_metrics.json",
            "summary_stats.json",
            f"p{int(QUANTILE_FOR_DEADLINE_MISS_RATE * 100)}_deadline_miss_rate_for_target_tbt_values.json",
        ]

        has_entries = False
        for relative_path in files_to_log:
            file_path = os.path.join(output_dir, relative_path)
            if os.path.exists(file_path):
                artifact.add_file(file_path, name=relative_path)
                has_entries = True

        if has_entries:
            wandb.log_artifact(artifact)

    def store_deadline_miss_rate_for_target_tbt(self, output_dir: str):
        # plot deadline miss rate for target TBT values
        deadline_miss_rate_for_target_tbt_values = (
            get_deadline_miss_rate_for_target_tbt_values(
                tbt_times=self.request_level_metrics.tbt,
                target_tbt_deadline_array=TARGET_TBT_RANGE,
                quantile=QUANTILE_FOR_DEADLINE_MISS_RATE,
            )
        )

        percentile_value = int(QUANTILE_FOR_DEADLINE_MISS_RATE * 100)
        x_axis_label = "Target TBT (ms)"
        y_axis_label = f"Miss Rate P({percentile_value})"

        data = {
            x_axis_label: [int(i * 1e3) for i in TARGET_TBT_RANGE],
            y_axis_label: deadline_miss_rate_for_target_tbt_values,
        }
        df = pd.DataFrame(data)

        with open(
            os.path.join(
                output_dir,
                f"p{percentile_value}_deadline_miss_rate_for_target_tbt_values.json",
            ),
            "w",
        ) as f:
            json.dump(data, f)

        if self.should_write_metrics_to_wandb and wandb.run:
            # plot deadline miss rate for target TBT values
            wandb.log(
                {
                    f"p{percentile_value}_deadline_miss_rate": wandb.plot.line(
                        table=wandb.Table(dataframe=df),
                        x=x_axis_label,
                        y=y_axis_label,
                        title="Deadline Miss Rate for Target TBT Values",
                    )
                },
            )

    def store_throughput_metrics(self, output_dir: str):
        (
            tpot_based_throughput,
            tbt_based_throughput,
            deadline_based_throughput,
        ) = get_throughput_metrics(
            self.request_level_metrics.tpot, self.request_level_metrics.tbt
        )

        throughput_metrics = {
            "tpot_based_throughput": tpot_based_throughput,
            "tbt_based_throughput": tbt_based_throughput,
            "deadline_based_throughput": deadline_based_throughput,
        }

        with open(os.path.join(output_dir, "throughput_metrics.json"), "w") as f:
            json.dump(throughput_metrics, f)

        # log plot of throughput metrics to wandb
        data = {
            "Metric Type": ["TPOT Based", "TBT Based", "Deadline Based"],
            "Throughput (tok/s)": [
                tpot_based_throughput,
                tbt_based_throughput,
                deadline_based_throughput,
            ],
        }
        df = pd.DataFrame(data)

        if self.should_write_metrics_to_wandb and wandb.run:
            wandb.log(
                {
                    "throughput_metrics": wandb.plot.bar(
                        table=wandb.Table(dataframe=df),
                        label="Metric Type",
                        value="Throughput (tok/s)",
                        title="Token Throughput",
                    )
                },
            )

    def store_ttft_violin_plots(self, output_dir: str):
        """Save and log TTFT distribution vs prompt length with a fixed, ordered x-axis.

        We bin prompt lengths into ordered ranges to avoid thousands of categories
        and to ensure the x-axis is monotonic.
        """
        if not self.request_level_metrics.ttft:
            return

        prompt_lengths = [int(n) for n in self.request_level_metrics.num_prompt_tokens]
        ttfts = list(self.request_level_metrics.ttft)
        if len(prompt_lengths) != len(ttfts):
            return

        base_df = pd.DataFrame({"prompt_length": prompt_lengths, "ttft": ttfts})

        # Choose a reasonable number of bins based on data spread, capped to 20
        min_len = int(base_df["prompt_length"].min())
        max_len = int(base_df["prompt_length"].max())
        if max_len <= min_len:
            # Degenerate case: single prompt length, no binning needed
            base_df["prompt_length_bin"] = pd.Series(
                [f"{min_len}–{max_len}"] * len(base_df)
            ).astype("category")
        else:
            # Target about 12 bins, but adapt to span to avoid overly dense ticks
            target_bins = 12
            bins = max(5, min(20, target_bins))
            # Evenly spaced edges -> coerce to unique integer edges to avoid duplicate labels
            raw_edges = np.linspace(min_len, max_len, bins + 1)
            int_edges = np.unique(np.round(raw_edges).astype(int))
            if int_edges.size < 2:
                base_df["prompt_length_bin"] = pd.Series(
                    [f"{min_len}–{max_len}"] * len(base_df)
                ).astype("category")
            else:
                edges = int_edges
                bins = edges.size - 1
                labels = [f"{edges[i]}–{edges[i + 1]}" for i in range(bins)]
                base_df["prompt_length_bin"] = pd.cut(
                    base_df["prompt_length"],
                    bins=edges,
                    include_lowest=True,
                    labels=labels,
                    right=True,
                )

        df = base_df[["prompt_length_bin", "ttft"]].copy()
        # Ensure the categorical keeps its intrinsic left-to-right ordering
        df["prompt_length_bin"] = df["prompt_length_bin"].astype("category")

        # Decide scale for TTFT; use native axis scaling to keep original ticks
        ttft_scale = recommend_axis_scale(df["ttft"], kind="numeric")
        y_label_linear = "TTFT (s)"
        y_label_scaled = (
            format_axis_label("TTFT", "s", ttft_scale)
            if ttft_scale != "linear"
            else y_label_linear
        )

        # 1) Save linear version
        fig_linear = rk.box(
            df,
            x="prompt_length_bin",
            y="ttft",
            labels={
                "prompt_length_bin": "Number of Prompt Tokens",
                "ttft": y_label_linear,
            },
        )
        fig_linear.save(os.path.join(output_dir, "ttft_violin_plot.png"))

        # 2) If scaled, also save a log/symlog variant
        if ttft_scale != "linear":
            fig_scaled = rk.box(
                df,
                x="prompt_length_bin",
                y="ttft",
                labels={
                    "prompt_length_bin": "Number of Prompt Tokens",
                    "ttft": y_label_scaled,
                },
            )
            apply_axis_scale(fig_scaled, axis="y", scale=ttft_scale)
            suffix = "log" if ttft_scale == "log" else "symlog"
            fig_scaled.save(
                os.path.join(output_dir, f"ttft_violin_plot_{suffix}_y.png")
            )
        if self.should_write_metrics_to_wandb and wandb.run:
            wandb.log(
                {
                    "ttft_violin_plot": wandb.Image(
                        os.path.join(output_dir, "ttft_violin_plot.png")
                    )
                }
            )
            wandb.log({"ttft_violin_data": wandb.Table(dataframe=df)})

    def store_generation_stalls(self, output_dir: str, request_idx: int = 0):
        # just generate for 1 request for now
        if request_idx >= len(self.request_level_metrics.ttft):
            return
        token_generated_times = [
            self.request_level_metrics.ttft[request_idx]
        ] + self.request_level_metrics.tbt[request_idx]
        token_generated_times = list(accumulate(token_generated_times))
        tokens_generated = list(range(1, len(token_generated_times) + 1))
        data = {
            "Time (s)": token_generated_times,
            "Tokens Generated": tokens_generated,
        }
        # Decide scale for time axis; set native axis scaling
        time_scale = recommend_axis_scale(token_generated_times, kind="numeric")
        x_label_linear = "Time (s)"
        x_label_scaled = (
            format_axis_label("Time", "s", time_scale)
            if time_scale != "linear"
            else x_label_linear
        )
        plot_df_linear = pd.DataFrame(
            {
                x_label_linear: token_generated_times,
                "Tokens Generated": tokens_generated,
            }
        )
        fig_linear = rk.line(
            plot_df_linear,
            x=x_label_linear,
            y="Tokens Generated",
            title="Tokens Generated vs Time",
        )
        fig_linear.save(os.path.join(output_dir, "tokens_generated_vs_time.png"))
        # Scaled variant if needed
        if time_scale != "linear":
            plot_df_scaled = pd.DataFrame(
                {
                    x_label_scaled: token_generated_times,
                    "Tokens Generated": tokens_generated,
                }
            )
            fig_scaled = rk.line(
                plot_df_scaled,
                x=x_label_scaled,
                y="Tokens Generated",
                title="Tokens Generated vs Time",
            )
            apply_axis_scale(fig_scaled, axis="x", scale=time_scale)
            suffix = "log" if time_scale == "log" else "symlog"
            fig_scaled.save(
                os.path.join(output_dir, f"tokens_generated_vs_time_{suffix}_x.png")
            )
        if self.should_write_metrics_to_wandb and wandb.run:
            wandb.log(
                {
                    "tokens_generated_vs_time": wandb.Image(
                        os.path.join(output_dir, "tokens_generated_vs_time.png")
                    )
                }
            )

    def store_summary_stats_table(self, output_dir: str) -> None:
        """Create and log a wandb table from summary_stats.json.

        Args:
            output_dir: Directory where summary_stats.json is written.
        """
        try:
            if not (self.should_write_metrics_to_wandb and wandb.run):
                return
            summary_json_path = os.path.join(output_dir, "summary_stats.json")
            if not os.path.exists(summary_json_path):
                return
            with open(summary_json_path, "r") as f:
                summary_dict = json.load(f)
            # Log only numeric metrics to avoid mixed-type column issues
            numeric_rows = []
            for key, value in summary_dict.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_rows.append({"Metric": key, "Value": float(value)})
            if numeric_rows:
                df = pd.DataFrame.from_records(numeric_rows)
                df = df[["Metric", "Value"]]
                wandb.log({"summary_stats_table": wandb.Table(dataframe=df)})

            # Optionally log error_code_freq as a separate table if present
            error_code_freq = summary_dict.get("error_code_freq")
            if isinstance(error_code_freq, dict) and len(error_code_freq) > 0:
                ec_rows = [
                    {"Error Code": str(code), "Count": int(count)}
                    for code, count in error_code_freq.items()
                ]
                ec_df = pd.DataFrame.from_records(ec_rows)
                wandb.log({"error_code_freq_table": wandb.Table(dataframe=ec_df)})
        except Exception as e:
            logger.warning(f"Failed to log summary_stats table to wandb: {e}")

    def _log_single_value_bar_chart(self, title: str, label: str, value: float) -> None:
        """Log a one-bar chart to wandb for a single scalar value."""
        if not (self.should_write_metrics_to_wandb and wandb.run):
            return
        try:
            df = pd.DataFrame({"Label": [label], "Value": [value]})
            wandb.log(
                {
                    title: wandb.plot.bar(
                        table=wandb.Table(dataframe=df),
                        label="Label",
                        value="Value",
                        title=title,
                    )
                },
            )
        except Exception as e:
            logger.warning(f"Failed to log bar chart '{title}': {e}")

    def store_ttft_tbt_scalar_charts(self) -> None:
        """Log single-value charts for ttft and tbt: min, mean, median, p90, p99, max."""
        if not (self.should_write_metrics_to_wandb and wandb.run):
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
            except Exception as e:
                logger.warning(f"Failed to compute stats for {short_name}: {e}")
                return

            for stat_name, stat_value in stats.items():
                title = f"{short_name} {stat_name}"
                self._log_single_value_bar_chart(
                    title=title,
                    label=short_name,
                    value=float(stat_value),
                )

        # TTFT (s) and TBT (s)
        log_for_sketch("ttft", "TTFT (s)")
        log_for_sketch("tbt", "TBT (s)")
