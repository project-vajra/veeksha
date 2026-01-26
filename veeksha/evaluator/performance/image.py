import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from veeksha.config.evaluator import (
    ImageChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.evaluator.base import EvaluationResult
from veeksha.evaluator.cdf_sketch import CDFSketch
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

logger = init_logger(__name__)


@dataclass
class ImageMetrics:
    """Metrics for single Image request."""

    request_id: int
    session_id: int
    dispatched_at: float
    completed_at: float
    num_prompt_tokens: int
    num_generated_images: int
    inter_chunk_times: List[float]
    is_stream: bool
    num_requested_images: int
    num_delta_prompt_tokens: int
    session_total_requests: Optional[int] = None

    @property
    def end_to_end_latency(self) -> float:
        return sum(self.inter_chunk_times)

    # Latency per generated image
    @property
    def latency_per_image(self) -> Optional[float]:
        if self.num_generated_images == 0:
            return None
        return self.end_to_end_latency / self.num_generated_images

    @property
    def generation_rate(self) -> Optional[float]:
        """Images generated per second for this request."""
        if self.end_to_end_latency == 0:
            return None
        return self.num_generated_images / self.end_to_end_latency


class ImagePerformanceEvaluator:
    """Performance evaluator for image generation (skeleton).

    -CDFSketch based metric aggregation
    -Request Level metrics tracking
    -Session Level metrics tracking
    -Throughput metrics
    -Output storing of images
    -WandB integration
    """

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        channel_config: Optional[ImageChannelPerformanceConfig] = None,
        benchmark_start_time: float = 0.0,
    ):
        self.config = config
        self.channel_config = channel_config or ImageChannelPerformanceConfig()
        self.benchmark_start_time = benchmark_start_time

        self.lock = threading.Lock()

        self._pending_requests: Dict[int, Dict[str, Any]] = (
            {}
        )  # request_id -> dispatch info

        # aggregated metrics
        self.summaries: Dict[str, CDFSketch] = {
            "num_prompt_tokens": CDFSketch(metric_name="num_prompt_tokens"),
            "num_generated_images": CDFSketch(metric_name="num_generated_images"),
            "end_to_end_latency": CDFSketch(metric_name="end_to_end_latency", unit="s"),
            "latency_per_image": CDFSketch(
                metric_name="latency_per_image", unit="s/image"
            ),
            "generation_rate": CDFSketch(
                metric_name="generation_rate", unit="images/s"
            ),
            "session_size": CDFSketch(metric_name="Requests per session"),
            "session_duration": CDFSketch(metric_name="Session duration", unit="s"),
            "session_think_time": CDFSketch(metric_name="Session think time", unit="s"),
        }

        # request-level metrics
        self._request_level_summary_keys = {
            "num_prompt_tokens",
            "num_generated_images",
            "end_to_end_latency",
            "latency_per_image",
            "generation_rate",
        }
        self.request_dispatched_at: List[float] = []
        self.completed_at: List[float] = []
        self.num_prompt_tokens: List[int] = []
        self.num_generated_images: List[int] = []
        self.num_requested_images: List[int] = []
        self.num_delta_prompt_tokens: List[int] = []
        self.end_to_end_latency: List[float] = []
        self.latency_per_image: List[Optional[float]] = []
        self.generation_rate: List[Optional[float]] = []
        self.session_ids: List[Optional[int]] = []
        self.session_total_requests: List[Optional[int]] = []
        self.request_ids: List[int] = []

        # Lifecycle timestamps
        self.scheduler_ready_at: List[Optional[float]] = []
        self.scheduler_dispatched_at: List[Optional[float]] = []
        self.client_picked_up_at: List[Optional[float]] = []
        self.client_completed_at: List[Optional[float]] = []
        self.result_processed_at: List[Optional[float]] = []

        # Streaming
        self.is_stream: List[bool] = []
        self._request_rows_streamed: int = 0
        self._request_time_reference = self.benchmark_start_time

        # session tracking
        self._session_last_completion: Dict[int, float] = {}

        # Image storage
        self.images: Dict[int, List[Any]] = {}

    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        content: Any,
        requested_output: Any = None,
    ) -> None:
        """Register an image request that was dispatched."""
        with self.lock:
            if self._request_time_reference == 0.0:
                self._request_time_reference = dispatched_at

            num_requested_images = 1
            if requested_output is not None and requested_output.image is not None:
                num_requested_images = requested_output.image.num_images

            self._pending_requests[request_id] = {
                "session_id": session_id,
                "dispatched_at": dispatched_at,
                "num_requested_images": num_requested_images,
            }
            logger.debug(
                f"Registered image request {request_id}, pending count: {len(self._pending_requests)}"
            )

    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
    ) -> None:
        """Record that an image request completed."""
        with self.lock:
            logger.debug(
                f"Completing image request {request_id}, pending count before: {len(self._pending_requests)}, pending IDs: {list(self._pending_requests.keys())}"
            )
            dispatch_info = self._pending_requests.pop(request_id, None)
            if dispatch_info is None:
                logger.warning(
                    f"Request {request_id} completed but was not registered. Current pending: {list(self._pending_requests.keys())}"
                )
                return

            dispatched_at = dispatch_info["dispatched_at"]
            num_requested_images = dispatch_info["num_requested_images"]

            # Extract metrics from channel response
            channel_response = response.channels.get(ChannelModality.IMAGE)
            if channel_response is not None:
                channel_metrics = channel_response.metrics or {}
                num_prompt_tokens = channel_metrics.get("num_total_prompt_tokens", 0)
                num_generated_images = channel_metrics.get("num_output_images", 0)
                inter_chunk_times = channel_metrics.get("inter_chunk_times", [])
                is_stream = channel_metrics.get("is_stream", False)
                num_delta_prompt_tokens = channel_metrics.get(
                    "num_delta_prompt_tokens", 0
                )
            else:
                num_prompt_tokens = 0
                num_generated_images = 0
                inter_chunk_times = []
                is_stream = False
                num_delta_prompt_tokens = 0
            session_total_requests = getattr(response, "session_total_requests", None)

            # Create metrics object
            metrics = ImageMetrics(
                request_id=request_id,
                session_id=session_id,
                dispatched_at=dispatched_at,
                completed_at=completed_at,
                num_prompt_tokens=num_prompt_tokens,
                num_generated_images=num_generated_images,
                inter_chunk_times=inter_chunk_times,
                is_stream=is_stream,
                num_requested_images=num_requested_images,
                session_total_requests=session_total_requests,
                num_delta_prompt_tokens=num_delta_prompt_tokens,
            )

            prev_completion = self._session_last_completion.get(session_id)
            if prev_completion is not None:
                think_time = dispatched_at - prev_completion
                if think_time >= 0:
                    self.summaries["session_think_time"].put(think_time)
            self._session_last_completion[session_id] = completed_at

            # Update CDF sketches
            self._update_summaries(metrics)

            # Store request-level metrics (including lifecycle timestamps from response)
            self._store_request_metrics(metrics, dispatched_at, response)

    def _update_summaries(self, metrics: ImageMetrics) -> None:
        """Update aggregated CDF sketches with new metrics."""
        for metric_name in self._request_level_summary_keys:
            value = getattr(metrics, metric_name, None)
            if value is not None:
                self.summaries[metric_name].put(value)

    def _store_request_metrics(
        self,
        metrics: ImageMetrics,
        dispatched_at: float,
        response: Any,
    ) -> None:
        """Store request-level metrics."""
        normalized_dispatched_at = max(
            0.0, dispatched_at - self._request_time_reference
        )
        self.request_dispatched_at.append(normalized_dispatched_at)
        self.completed_at.append(
            max(0.0, metrics.completed_at - self._request_time_reference)
        )
        self.num_prompt_tokens.append(metrics.num_prompt_tokens)
        self.num_generated_images.append(metrics.num_generated_images)
        self.num_requested_images.append(metrics.num_requested_images)
        self.num_delta_prompt_tokens.append(metrics.num_delta_prompt_tokens)
        self.end_to_end_latency.append(metrics.end_to_end_latency)
        self.latency_per_image.append(metrics.latency_per_image)
        self.generation_rate.append(metrics.generation_rate)
        self.session_ids.append(metrics.session_id)
        self.session_total_requests.append(metrics.session_total_requests)
        self.request_ids.append(metrics.request_id)
        self.is_stream.append(metrics.is_stream)

        imgs = (
            response.channels.get(ChannelModality.IMAGE).content
            if response.channels.get(ChannelModality.IMAGE)
            else []
        )

        if self.channel_config and getattr(self.channel_config, "save_images", False):
            self.images[metrics.request_id] = imgs

        # Extract and store lifecycle timestamps from response
        def normalize_ts(ts: Optional[float]) -> Optional[float]:
            if ts is None:
                return None
            return max(0.0, ts - self._request_time_reference)

        self.scheduler_ready_at.append(
            normalize_ts(getattr(response, "scheduler_ready_at", None))
        )

        self.scheduler_dispatched_at.append(
            normalize_ts(getattr(response, "scheduler_dispatched_at", None))
        )

        self.client_picked_up_at.append(
            normalize_ts(getattr(response, "client_picked_up_at", None))
        )

        self.client_completed_at.append(
            normalize_ts(getattr(response, "client_completed_at", None))
        )
        self.result_processed_at.append(
            normalize_ts(getattr(response, "result_processed_at", None))
        )

    def record_session_completed(
        self,
        session_id: int,
        session_size: int,
        first_dispatch_at: Optional[float],
        last_completion_at: Optional[float],
    ) -> None:
        """Record session-level metrics."""
        with self.lock:
            self.summaries["session_size"].put(session_size)

            # Session duration
            if first_dispatch_at is not None and last_completion_at is not None:
                duration = max(0.0, last_completion_at - first_dispatch_at)
                self.summaries["session_duration"].put(duration)

            # Clean up session state
            self._session_last_completion.pop(session_id, None)

    def get_summary(self) -> Dict[str, float]:
        """Get summary metrics from all CDF sketches."""
        perf_summary = {}
        for cdf_sketch in self.summaries.values():
            perf_summary.update(cdf_sketch.get_summary())

        return perf_summary

    def finalize(self) -> EvaluationResult:
        """Finalize evaluation and return results."""
        with self.lock:
            return EvaluationResult(
                evaluator_type="image_performance",
                channel=ChannelModality.IMAGE,
                metrics=self.get_summary(),
            )

    def get_streaming_metrics(self) -> Optional[Dict[str, Any]]:
        """Return current metrics for streaming."""
        with self.lock:
            return self.get_summary()

    def save(self, output_dir: str) -> None:
        """Save evaluation artifacts."""
        with self.lock:
            self._save_request_level_metrics(output_dir)
            self._save_cdf_csvs(output_dir)
            self._save_images(output_dir)
            self._save_throughput_metrics(output_dir)
            self._plot_cdfs(output_dir)
            self._log_wandb_metrics(output_dir)

    def flush_streaming_outputs(self, output_dir: str) -> None:
        """Flush current metrics for streaming."""
        with self.lock:
            rows = self._export_request_rows(self._request_rows_streamed)
            if rows:
                self._append_request_level_rows(output_dir, rows)
                self._request_rows_streamed += len(rows)

            # Save current CDF summaries
            self._save_cdf_csvs(output_dir)

    # Output saving methods
    def _save_request_level_metrics(self, output_dir: str) -> None:
        """Save request-level metrics as JSONL."""
        path = os.path.join(output_dir, "request_level_metrics.jsonl")
        rows = self._export_request_rows(0)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row))
                f.write("\n")

    def _export_request_rows(self, start_index: int) -> List[Dict[str, Any]]:
        """Export request-level metrics as list of dicts."""
        rows = []
        num_requests = len(self.request_ids)
        for i in range(start_index, num_requests):
            row = {
                "request_id": self.request_ids[i],
                "session_id": self.session_ids[i],
                "dispatched_at": self.request_dispatched_at[i],
                "completed_at": self.completed_at[i],
                "num_prompt_tokens": self.num_prompt_tokens[i],
                "num_generated_images": self.num_generated_images[i],
                "num_requested_images": self.num_requested_images[i],
                "num_delta_prompt_tokens": (
                    self.num_delta_prompt_tokens[i]
                    if i < len(self.num_delta_prompt_tokens)
                    else None
                ),
                "end_to_end_latency": (
                    round(self.end_to_end_latency[i], 5)
                    if self.end_to_end_latency[i] is not None  # type: ignore[arg-type]
                    else None
                ),
                "latency_per_image": (
                    round(self.latency_per_image[i], 5)  # type: ignore[arg-type]
                    if self.latency_per_image[i] is not None
                    else None
                ),
                "generation_rate": (
                    round(self.generation_rate[i], 5)  # type: ignore[arg-type]
                    if self.generation_rate[i] is not None
                    else None
                ),
                "session_total_requests": self.session_total_requests[i],
                "scheduler_ready_at": self.scheduler_ready_at[i],
                "scheduler_dispatched_at": self.scheduler_dispatched_at[i],
                "client_picked_up_at": self.client_picked_up_at[i],
                "client_completed_at": self.client_completed_at[i],
                "result_processed_at": self.result_processed_at[i],
                "is_stream": self.is_stream[i],
            }
            rows.append(row)
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

    def _save_images(self, output_dir: str) -> None:
        """Save generated images to output directory."""

        if not getattr(self.channel_config, "save_images", False):
            return

        out_dir = os.path.join(output_dir, "generated_images")
        os.makedirs(out_dir, exist_ok=True)

        for request_id, imgs in self.images.items():
            for idx, img in enumerate(imgs):
                if isinstance(img, (bytes, bytearray)):

                    img_path = os.path.join(
                        out_dir, f"request_{request_id}_img_{idx}.png"
                    )
                    with open(img_path, "wb") as f:
                        f.write(img)
                else:
                    logger.warning(
                        f"Image for request {request_id}, index {idx} is not in bytes format. Skipping save."
                    )

    def _plot_cdfs(self, output_dir: str) -> None:
        """Generate CDF plots for all metrics."""
        for metric_name, cdf_sketch in self.summaries.items():
            cdf_sketch.plot_cdf(output_dir, metric_name)

    def _save_throughput_metrics(self, output_dir: str) -> None:
        """Save throughput metrics (non-streaming, E2E based only)."""
        # System-level throughput: total images / total time
        total_images = sum(self.num_generated_images)
        if self.request_dispatched_at and self.completed_at:
            total_time = max(self.completed_at) - min(self.request_dispatched_at)
            system_throughput = total_images / total_time if total_time > 0 else 0.0
        else:
            system_throughput = 0.0

        metrics = {
            "system_throughput_images_per_sec": system_throughput,
            "total_images_generated": total_images,
        }
        path = os.path.join(output_dir, "throughput_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    def _log_wandb_metrics(self, output_dir: str) -> None:
        try:
            from typing import Any, cast

            import wandb  # type: ignore[import-not-found]

            wandb = cast(Any, wandb)

            if not getattr(wandb, "run", None):
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
                    "Metric": ["System Throughput"],
                    "Images/sec": [
                        throughput.get("system_throughput_images_per_sec", 0.0)
                    ],
                }
                df = pd.DataFrame(data)
                wandb.log(
                    {
                        "throughput_metrics": wandb.plot.bar(
                            table=wandb.Table(dataframe=df),
                            label="Metric",
                            value="Images/sec",
                            title="Image Generation Throughput",
                        )
                    }
                )

        except Exception as e:
            logger.warning(f"Failed to log WandB metrics: {e}")
