"""Performance evaluator for TTS, realtime TTS, LLM-audio, and STT metrics."""

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from veeksha.config.evaluator import (
    AudioChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.core.audio_contract import (
    DEFAULT_AUDIO_SAMPLE_RATE,
    WAV_HEADER_BYTES,
    AudioMetricKey,
    pcm_bytes_to_duration_ms,
)
from veeksha.evaluator.base import EvaluationResult
from veeksha.evaluator.cdf_sketch import CDFSketch
from veeksha.evaluator.performance.asr import (
    ASRMetricAccumulator,
    ASRRequestMetrics,
    score_asr_request,
)
from veeksha.evaluator.performance.audio_interactivity import (
    InteractivityMetrics,
    RequestTiming,
    compute_interactivity_metrics,
    parse_request_timing,
)
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

logger = init_logger(__name__)

# One codec chunk of audio in milliseconds. A stream truncated at the server's
# length cap can fall short of the exact cap duration by at most one chunk, so
# the truncation heuristic compares against (cap - one chunk).
LENGTH_CAP_CHUNK_MS = 320.0


def _pcm_byte_count(total_bytes: int, *, raw_pcm: bool) -> int:
    if raw_pcm:
        return total_bytes
    return max(total_bytes - WAV_HEADER_BYTES, 0)


def _policy_tag(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


@dataclass
class AudioRequestMetrics:
    """Metrics for a single audio request."""

    request_id: int
    session_id: int
    request_dispatched_at: float
    client_completed_at: float
    ttfc: float
    end_to_end_latency: float
    generated_audio_duration: float
    rtf: float
    chunk_count: int
    pcm_byte_count: int
    input_chars: int = 0
    audio_task: AudioTask | None = None
    asr: ASRRequestMetrics | None = None
    input_tokens: int = 0
    input_text: str = ""
    session_total_requests: int | None = None
    interactivity: InteractivityMetrics | None = None
    ws_connect_latency_ms: float | None = None
    suspected_length_cap_truncation: bool = False


class AudioPerformanceEvaluator:
    """Evaluate TTS interactivity and STT recognition performance together."""

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        channel_config: AudioChannelPerformanceConfig | None = None,
        benchmark_start_time: float = 0.0,
    ):
        self.config = config
        self.channel_config = channel_config or AudioChannelPerformanceConfig()
        self.benchmark_start_time = benchmark_start_time
        self.lock = threading.Lock()

        self.summaries: dict[str, CDFSketch] = {
            AudioMetricKey.TTFC.value: CDFSketch(AudioMetricKey.TTFC.value, unit="ms"),
            AudioMetricKey.END_TO_END_LATENCY.value: CDFSketch(
                AudioMetricKey.END_TO_END_LATENCY.value, unit="ms"
            ),
            AudioMetricKey.GENERATED_AUDIO_DURATION.value: CDFSketch(
                AudioMetricKey.GENERATED_AUDIO_DURATION.value, unit="ms"
            ),
            AudioMetricKey.RTF.value: CDFSketch(AudioMetricKey.RTF.value),
            AudioMetricKey.CHUNK_COUNT.value: CDFSketch(
                AudioMetricKey.CHUNK_COUNT.value
            ),
            AudioMetricKey.INPUT_TOKENS.value: CDFSketch(
                AudioMetricKey.INPUT_TOKENS.value, unit="tokens"
            ),
            AudioMetricKey.SESSION_SIZE.value: CDFSketch(
                AudioMetricKey.SESSION_SIZE.value
            ),
            AudioMetricKey.SESSION_DURATION.value: CDFSketch(
                AudioMetricKey.SESSION_DURATION.value, unit="ms"
            ),
        }

        self.asr_latency_summaries: dict[str, CDFSketch] = {
            "time_to_first_visible_text": CDFSketch(
                "time_to_first_visible_text", unit="ms"
            ),
            "time_to_first_partial": CDFSketch("time_to_first_partial", unit="ms"),
            "time_to_final_transcript": CDFSketch(
                "time_to_final_transcript", unit="ms"
            ),
            "interactivity": CDFSketch("interactivity", unit="ms"),
        }
        self._asr_metrics = ASRMetricAccumulator()
        self._stt_request_count = 0
        self._asr_scoring_seconds = 0.0

        self._pending_requests: dict[int, dict[str, Any]] = {}
        self._completed_metrics: list[AudioRequestMetrics] = []
        self._lifecycle_timestamps: list[dict[str, float | None]] = []
        self._request_rows_streamed = 0
        self._request_time_reference = self.benchmark_start_time

        self._interactivity_sketches_ready = False
        self._interactive_request_count = 0
        self._fixed_delay_stall_free_counts: dict[float, int] = {}
        self._buffer_target_stall_free_counts: dict[float, int] = {}
        self._buffer_target_eligible_counts: dict[float, int] = {}
        self._raw_timing_rows: list[dict[str, Any]] = []
        self._raw_timing_rows_streamed = 0

        self._total_input_chars = 0
        self._total_generated_audio_duration_ms = 0.0
        self._first_dispatch_at: float | None = None
        self._last_completion_at: float | None = None
        self._suspected_truncation_count = 0

    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        content: Any,
        requested_output: Any = None,
    ) -> None:
        """Register an audio request that was dispatched."""
        with self.lock:
            if self._request_time_reference == 0.0:
                self._request_time_reference = dispatched_at
            self._pending_requests[request_id] = {
                "session_id": session_id,
                "dispatched_at": dispatched_at,
            }

    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
    ) -> None:
        """Record a completed TTS, LLM-audio, or STT request.

        ASR scoring runs outside the evaluator lock so completion workers can
        normalize and align transcripts concurrently. Shared metric mutation
        remains under the evaluator lock.
        """
        with self.lock:
            dispatch_info = self._pending_requests.pop(request_id, None)
            request_time_reference = self._request_time_reference

        dispatched_at = (
            dispatch_info["dispatched_at"]
            if dispatch_info
            else getattr(
                response,
                "scheduler_dispatched_at",
                request_time_reference,
            )
        )

        channel_response = getattr(response, "channels", {}).get(ChannelModality.AUDIO)
        if channel_response is None:
            logger.debug("Request %d has no AUDIO channel response", request_id)
            return

        cm = channel_response.metrics or {}
        ttfc = float(cm.get(AudioMetricKey.TTFC.value, 0.0))
        end_to_end_latency = float(cm.get(AudioMetricKey.END_TO_END_LATENCY.value, 0.0))
        chunk_count = int(cm.get(AudioMetricKey.CHUNK_COUNT.value, 0))
        raw_pcm = bool(cm.get(AudioMetricKey.RAW_PCM.value, False))
        sample_rate = int(
            cm.get(AudioMetricKey.SAMPLE_RATE.value, DEFAULT_AUDIO_SAMPLE_RATE)
        )

        raw_audio_task = cm.get("audio_task")
        try:
            audio_task = (
                AudioTask[raw_audio_task.upper()]
                if isinstance(raw_audio_task, str)
                else AudioTask(raw_audio_task)
            )
        except (KeyError, TypeError, ValueError) as error:
            expected = ", ".join(str(task) for task in AudioTask)
            raise ValueError(
                f"AUDIO response for request {request_id} has unknown "
                f"audio_task={raw_audio_task!r}; expected one of: {expected}"
            ) from error

        if audio_task is AudioTask.STT:
            if AudioMetricKey.PCM_BYTE_COUNT.value not in cm:
                raise ValueError(
                    f"STT response for request {request_id} is missing "
                    f"{AudioMetricKey.PCM_BYTE_COUNT.value}"
                )
            total_bytes = int(cm[AudioMetricKey.PCM_BYTE_COUNT.value])
        else:
            audio_content = channel_response.content
            total_bytes = (
                len(audio_content)
                if isinstance(audio_content, (bytes, bytearray, memoryview))
                else 0
            )

        pcm_byte_count = _pcm_byte_count(total_bytes, raw_pcm=raw_pcm)
        generated_audio_duration = pcm_bytes_to_duration_ms(pcm_byte_count, sample_rate)
        rtf = (
            end_to_end_latency / generated_audio_duration
            if generated_audio_duration > 0
            else float("inf")
        )
        input_chars = int(cm.get(AudioMetricKey.INPUT_CHARS.value, 0) or 0)
        input_tokens = int(cm.get(AudioMetricKey.INPUT_TOKENS.value, 0) or 0)
        input_text = str(cm.get(AudioMetricKey.INPUT_TEXT.value, ""))

        # Duration at the server's length cap (within one codec chunk) is the
        # only client-visible signal of silent server-side truncation.
        max_expected_audio_ms = self.channel_config.max_expected_audio_ms
        suspected_length_cap_truncation = (
            audio_task is AudioTask.TTS
            and max_expected_audio_ms is not None
            and generated_audio_duration >= max_expected_audio_ms - LENGTH_CAP_CHUNK_MS
        )

        ws_connect_latency = cm.get(AudioMetricKey.WS_CONNECT_LATENCY_MS.value)
        ws_connect_latency_ms = (
            float(ws_connect_latency) if ws_connect_latency is not None else None
        )
        interactivity: InteractivityMetrics | None = None
        raw_timing_row: dict[str, Any] | None = None
        if audio_task is AudioTask.TTS and self.channel_config.interactivity_enabled:
            timing = parse_request_timing(cm, sample_rate)
            if timing is not None:
                interactivity = compute_interactivity_metrics(
                    timing,
                    startup_delay_ms_values=(
                        self.channel_config.startup_delay_ms_values
                    ),
                    startup_buffer_ms_values=(
                        self.channel_config.startup_buffer_ms_values
                    ),
                    min_reportable_stall_ms=(
                        self.channel_config.min_reportable_stall_ms
                    ),
                )
                if self.channel_config.persist_raw_timing:
                    raw_timing_row = self._build_raw_timing_row(
                        request_id, session_id, cm, timing
                    )

        asr_metrics: ASRRequestMetrics | None = None
        asr_scoring_seconds = 0.0
        if audio_task is AudioTask.STT:
            scoring_started_at = time.perf_counter()
            asr_metrics = score_asr_request(
                request_id=request_id,
                channel_metrics=cm,
                duration_s=generated_audio_duration / 1000.0,
                accumulator=self._asr_metrics,
            )
            asr_scoring_seconds = time.perf_counter() - scoring_started_at

        with self.lock:
            if audio_task is AudioTask.STT:
                self._stt_request_count += 1
                self._asr_scoring_seconds += asr_scoring_seconds
            if raw_timing_row is not None:
                self._raw_timing_rows.append(raw_timing_row)

            metrics = AudioRequestMetrics(
                request_id=request_id,
                session_id=session_id,
                request_dispatched_at=dispatched_at,
                client_completed_at=completed_at,
                ttfc=ttfc,
                end_to_end_latency=end_to_end_latency,
                generated_audio_duration=generated_audio_duration,
                rtf=rtf,
                chunk_count=chunk_count,
                pcm_byte_count=pcm_byte_count,
                input_chars=input_chars,
                input_tokens=input_tokens,
                audio_task=audio_task,
                asr=asr_metrics,
                input_text=input_text,
                session_total_requests=getattr(
                    response, "session_total_requests", None
                ),
                interactivity=interactivity,
                ws_connect_latency_ms=ws_connect_latency_ms,
                suspected_length_cap_truncation=suspected_length_cap_truncation,
            )
            self._completed_metrics.append(metrics)
            if suspected_length_cap_truncation:
                self._suspected_truncation_count += 1

            self._total_input_chars += input_chars
            self._total_generated_audio_duration_ms += generated_audio_duration
            if (
                self._first_dispatch_at is None
                or dispatched_at < self._first_dispatch_at
            ):
                self._first_dispatch_at = dispatched_at
            if (
                self._last_completion_at is None
                or completed_at > self._last_completion_at
            ):
                self._last_completion_at = completed_at

            def normalize_ts(timestamp: float | None) -> float | None:
                if timestamp is None:
                    return None
                return round(max(0.0, timestamp - self._request_time_reference), 5)

            self._lifecycle_timestamps.append(
                {
                    "scheduler_ready_at": normalize_ts(
                        getattr(response, "scheduler_ready_at", None)
                    ),
                    "scheduler_dispatched_at": normalize_ts(
                        getattr(response, "scheduler_dispatched_at", None)
                    ),
                    "client_picked_up_at": normalize_ts(
                        getattr(response, "client_picked_up_at", None)
                    ),
                    "client_completed_at": normalize_ts(
                        getattr(response, "client_completed_at", None)
                    ),
                    "result_processed_at": normalize_ts(
                        getattr(response, "result_processed_at", None)
                    ),
                }
            )

            self.summaries[AudioMetricKey.TTFC.value].put(ttfc)
            self.summaries[AudioMetricKey.END_TO_END_LATENCY.value].put(
                end_to_end_latency
            )
            self.summaries[AudioMetricKey.GENERATED_AUDIO_DURATION.value].put(
                generated_audio_duration
            )
            self.summaries[AudioMetricKey.RTF.value].put(rtf)
            self.summaries[AudioMetricKey.CHUNK_COUNT.value].put(chunk_count)
            self.summaries[AudioMetricKey.INPUT_TOKENS.value].put(input_tokens)

            if asr_metrics is not None:
                asr_values = {
                    "time_to_first_visible_text": (
                        asr_metrics.time_to_first_visible_text
                    ),
                    "time_to_first_partial": asr_metrics.time_to_first_partial,
                    "time_to_final_transcript": (asr_metrics.time_to_final_transcript),
                    "interactivity": asr_metrics.interactivity,
                }
                for metric_name, value in asr_values.items():
                    if value is not None:
                        self.asr_latency_summaries[metric_name].put(value)

            if interactivity is not None:
                self._record_interactivity(metrics)

    def _ensure_interactivity_sketches(self) -> None:
        if self._interactivity_sketches_ready:
            return
        specs: list[tuple[AudioMetricKey, str | None]] = [
            (AudioMetricKey.FIRST_INPUT_TO_FIRST_AUDIO_MS, "ms"),
            (AudioMetricKey.REQUEST_START_TO_FIRST_AUDIO_MS, "ms"),
            (AudioMetricKey.AUDIO_BEFORE_COMMIT_RATIO, None),
            (AudioMetricKey.POST_COMMIT_AUDIO_DELIVERY_MS, "ms"),
            (AudioMetricKey.REQUIRED_STARTUP_DELAY_MS, "ms"),
            (AudioMetricKey.ZERO_DELAY_STALL_COUNT, None),
            (AudioMetricKey.ZERO_DELAY_TOTAL_STALL_MS, "ms"),
            (AudioMetricKey.ZERO_DELAY_LONGEST_STALL_MS, "ms"),
            (AudioMetricKey.STREAMING_RTF, None),
            (AudioMetricKey.DONE_AFTER_LAST_AUDIO_MS, "ms"),
            (AudioMetricKey.WS_CONNECT_LATENCY_MS, "ms"),
        ]
        for key, unit in specs:
            self.summaries[key.value] = CDFSketch(key.value, unit=unit)
        self._interactivity_sketches_ready = True

    def _record_interactivity(self, metrics: AudioRequestMetrics) -> None:
        interactivity = metrics.interactivity
        if interactivity is None:
            return
        self._ensure_interactivity_sketches()

        def put(key: AudioMetricKey, value: float | None) -> None:
            if value is not None:
                self.summaries[key.value].put(value)

        zero_delay = interactivity.fixed_delay_playback[0.0]
        put(
            AudioMetricKey.FIRST_INPUT_TO_FIRST_AUDIO_MS,
            interactivity.first_input_to_first_audio_ms,
        )
        put(
            AudioMetricKey.REQUEST_START_TO_FIRST_AUDIO_MS,
            interactivity.request_start_to_first_audio_ms,
        )
        put(
            AudioMetricKey.AUDIO_BEFORE_COMMIT_RATIO,
            interactivity.audio_before_commit_ratio,
        )
        put(
            AudioMetricKey.POST_COMMIT_AUDIO_DELIVERY_MS,
            interactivity.post_commit_audio_delivery_ms,
        )
        put(
            AudioMetricKey.REQUIRED_STARTUP_DELAY_MS,
            interactivity.required_startup_delay_ms,
        )
        put(AudioMetricKey.ZERO_DELAY_STALL_COUNT, float(zero_delay.stall_count))
        put(AudioMetricKey.ZERO_DELAY_TOTAL_STALL_MS, zero_delay.total_stall_ms)
        put(AudioMetricKey.ZERO_DELAY_LONGEST_STALL_MS, zero_delay.longest_stall_ms)
        put(AudioMetricKey.STREAMING_RTF, interactivity.streaming_rtf)
        put(
            AudioMetricKey.DONE_AFTER_LAST_AUDIO_MS,
            interactivity.done_after_last_audio_ms,
        )
        put(AudioMetricKey.WS_CONNECT_LATENCY_MS, metrics.ws_connect_latency_ms)

        self._interactive_request_count += 1
        for delay_ms, result in interactivity.fixed_delay_playback.items():
            if result.stall_free:
                self._fixed_delay_stall_free_counts[delay_ms] = (
                    self._fixed_delay_stall_free_counts.get(delay_ms, 0) + 1
                )
        for target_ms, result in interactivity.buffer_target_playback.items():
            if result is None:
                continue
            self._buffer_target_eligible_counts[target_ms] = (
                self._buffer_target_eligible_counts.get(target_ms, 0) + 1
            )
            if result.stall_free:
                self._buffer_target_stall_free_counts[target_ms] = (
                    self._buffer_target_stall_free_counts.get(target_ms, 0) + 1
                )

    def record_session_completed(
        self,
        session_id: int,
        session_size: int,
        first_dispatch_at: float | None,
        last_completion_at: float | None,
    ) -> None:
        """Record session-level metrics."""
        with self.lock:
            self.summaries[AudioMetricKey.SESSION_SIZE.value].put(session_size)
            if first_dispatch_at is not None and last_completion_at is not None:
                self.summaries[AudioMetricKey.SESSION_DURATION.value].put(
                    max(0.0, last_completion_at - first_dispatch_at)
                )

    def get_summary(self) -> dict[str, float | None]:
        """Return aggregate audio, ASR, and playback-policy metrics."""
        perf_summary: dict[str, float | None] = {}
        for cdf_sketch in self.summaries.values():
            perf_summary.update(cdf_sketch.get_summary())

        if self._stt_request_count > 0:
            for cdf_sketch in self.asr_latency_summaries.values():
                if len(cdf_sketch) > 0:
                    perf_summary.update(cdf_sketch.get_summary())
            perf_summary.update(self._asr_metrics.get_summary())

        wall_s = 0.0
        if self._first_dispatch_at is not None and self._last_completion_at is not None:
            wall_s = max(0.0, self._last_completion_at - self._first_dispatch_at)
        perf_summary["chars_per_sec_aggregate"] = (
            self._total_input_chars / wall_s if wall_s > 0 else None
        )
        perf_summary["generated_audio_seconds_per_sec_aggregate"] = (
            (self._total_generated_audio_duration_ms / 1000.0) / wall_s
            if wall_s > 0
            else None
        )

        if self.channel_config.max_expected_audio_ms is not None:
            perf_summary["suspected_length_cap_truncation"] = (
                self._suspected_truncation_count
            )

        if self._interactive_request_count > 0:
            count = self._interactive_request_count
            perf_summary["interactive_requests_count"] = count
            for delay_ms in self.channel_config.startup_delay_ms_values:
                tag = _policy_tag(delay_ms)
                fraction = self._fixed_delay_stall_free_counts.get(delay_ms, 0) / count
                perf_summary[f"fixed_delay_stall_free_fraction_d{tag}ms"] = fraction
                if delay_ms == 0.0:
                    perf_summary["zero_delay_stall_free_fraction"] = fraction
            for target_ms in self.channel_config.startup_buffer_ms_values:
                tag = _policy_tag(target_ms)
                eligible = self._buffer_target_eligible_counts.get(target_ms, 0)
                perf_summary[f"buffer_target_eligible_count_b{tag}ms"] = eligible
                perf_summary[f"buffer_target_stall_free_fraction_b{tag}ms"] = (
                    self._buffer_target_stall_free_counts.get(target_ms, 0) / eligible
                    if eligible > 0
                    else None
                )
        return perf_summary

    def finalize(self) -> EvaluationResult:
        with self.lock:
            if self._stt_request_count > 0:
                logger.info(
                    "Evaluator phase 'asr_scoring' took %.2fs total across "
                    "%d STT requests (concurrent and overlapped with the run)",
                    self._asr_scoring_seconds,
                    self._stt_request_count,
                )
            return EvaluationResult(
                evaluator_type="audio_performance",
                channel=ChannelModality.AUDIO,
                metrics=self.get_summary(),
            )

    def get_streaming_metrics(self) -> dict[str, Any] | None:
        with self.lock:
            return self.get_summary()

    def save(self, output_dir: str) -> None:
        with self.lock:
            stages = (
                ("audio_request_level_metrics", self._save_request_level_metrics),
                ("audio_raw_timing", self._save_raw_timing),
                ("audio_cdf_csvs", self._save_cdf_csvs),
                ("audio_cdf_plots", self._plot_cdfs),
                (
                    "audio_stall_free_policy_plot",
                    self._plot_stall_free_vs_startup_policy,
                ),
            )
            for stage_name, stage in stages:
                stage_started_at = time.perf_counter()
                stage(output_dir)
                logger.info(
                    "Evaluator phase '%s' took %.2fs",
                    stage_name,
                    time.perf_counter() - stage_started_at,
                )

    def flush_streaming_outputs(self, output_dir: str) -> None:
        with self.lock:
            rows = self._export_request_rows(self._request_rows_streamed)
            if rows:
                self._append_request_level_rows(output_dir, rows)
                self._request_rows_streamed = len(self._completed_metrics)
            self._flush_raw_timing_rows(output_dir)
            self._save_cdf_csvs(output_dir)

    def _save_request_level_metrics(self, output_dir: str) -> None:
        path = os.path.join(output_dir, "request_level_metrics.jsonl")
        with open(path, "w") as file:
            for row in self._export_request_rows(0):
                file.write(json.dumps(row) + "\n")

    def _export_request_rows(self, start_index: int = 0) -> list[dict[str, Any]]:
        rows = []
        for index in range(start_index, len(self._completed_metrics)):
            metrics = self._completed_metrics[index]
            lifecycle = self._lifecycle_timestamps[index]
            row = {
                "request_id": metrics.request_id,
                "session_id": metrics.session_id,
                "session_total_requests": metrics.session_total_requests,
                "scheduler_ready_at": lifecycle["scheduler_ready_at"],
                "scheduler_dispatched_at": round(
                    max(
                        0.0,
                        metrics.request_dispatched_at - self._request_time_reference,
                    ),
                    5,
                ),
                "client_picked_up_at": lifecycle["client_picked_up_at"],
                "client_completed_at": round(
                    max(
                        0.0,
                        metrics.client_completed_at - self._request_time_reference,
                    ),
                    5,
                ),
                "result_processed_at": lifecycle["result_processed_at"],
                "audio_task": (
                    str(metrics.audio_task) if metrics.audio_task is not None else None
                ),
                AudioMetricKey.TTFC.value: round(metrics.ttfc, 3),
                AudioMetricKey.END_TO_END_LATENCY.value: round(
                    metrics.end_to_end_latency, 3
                ),
                AudioMetricKey.GENERATED_AUDIO_DURATION.value: round(
                    metrics.generated_audio_duration, 3
                ),
                AudioMetricKey.RTF.value: round(metrics.rtf, 5),
                AudioMetricKey.CHUNK_COUNT.value: metrics.chunk_count,
                AudioMetricKey.PCM_BYTE_COUNT.value: metrics.pcm_byte_count,
                AudioMetricKey.INPUT_CHARS.value: metrics.input_chars,
                AudioMetricKey.INPUT_TOKENS.value: metrics.input_tokens,
                AudioMetricKey.INPUT_TEXT.value: metrics.input_text,
            }
            if self.channel_config.max_expected_audio_ms is not None:
                row["suspected_length_cap_truncation"] = int(
                    metrics.suspected_length_cap_truncation
                )
            if metrics.asr is not None:
                row.update(metrics.asr.to_request_row())
            row.update(self._interactivity_row_fields(metrics))
            rows.append(row)
        return rows

    def _interactivity_row_fields(self, metrics: AudioRequestMetrics) -> dict[str, Any]:
        interactivity = metrics.interactivity
        if interactivity is None:
            return {}

        fields: dict[str, Any] = {}

        def add(key: str, value: Any, ndigits: int = 3) -> None:
            if value is None:
                return
            fields[key] = round(value, ndigits) if isinstance(value, float) else value

        zero_delay = interactivity.fixed_delay_playback[0.0]
        add(
            AudioMetricKey.FIRST_INPUT_TO_FIRST_AUDIO_MS.value,
            interactivity.first_input_to_first_audio_ms,
        )
        add(
            AudioMetricKey.REQUEST_START_TO_FIRST_AUDIO_MS.value,
            interactivity.request_start_to_first_audio_ms,
        )
        add(
            AudioMetricKey.AUDIO_BEFORE_COMMIT_RATIO.value,
            interactivity.audio_before_commit_ratio,
            5,
        )
        add(
            AudioMetricKey.POST_COMMIT_AUDIO_DELIVERY_MS.value,
            interactivity.post_commit_audio_delivery_ms,
        )
        add(
            AudioMetricKey.REQUIRED_STARTUP_DELAY_MS.value,
            interactivity.required_startup_delay_ms,
        )
        fields[AudioMetricKey.ZERO_DELAY_STALL_COUNT.value] = zero_delay.stall_count
        add(
            AudioMetricKey.ZERO_DELAY_TOTAL_STALL_MS.value,
            zero_delay.total_stall_ms,
        )
        add(
            AudioMetricKey.ZERO_DELAY_LONGEST_STALL_MS.value,
            zero_delay.longest_stall_ms,
        )
        fields[AudioMetricKey.ZERO_DELAY_STALL_FREE.value] = int(zero_delay.stall_free)
        add(AudioMetricKey.STREAMING_RTF.value, interactivity.streaming_rtf, 5)
        add(
            AudioMetricKey.DONE_AFTER_LAST_AUDIO_MS.value,
            interactivity.done_after_last_audio_ms,
        )

        for delay_ms, result in interactivity.fixed_delay_playback.items():
            if delay_ms == 0.0:
                continue
            prefix = f"fixed_delay_{_policy_tag(delay_ms)}ms"
            fields[f"{prefix}_stall_count"] = result.stall_count
            add(f"{prefix}_total_stall_ms", result.total_stall_ms)
            add(f"{prefix}_longest_stall_ms", result.longest_stall_ms)
            fields[f"{prefix}_stall_free"] = int(result.stall_free)

        for target_ms, result in interactivity.buffer_target_playback.items():
            if result is None:
                continue
            prefix = f"buffer_target_{_policy_tag(target_ms)}ms"
            add(f"{prefix}_startup_wait_ms", result.startup_wait_from_first_audio_ms)
            fields[f"{prefix}_stall_count"] = result.stall_count
            add(f"{prefix}_total_stall_ms", result.total_stall_ms)
            add(f"{prefix}_longest_stall_ms", result.longest_stall_ms)
            fields[f"{prefix}_stall_free"] = int(result.stall_free)
        return fields

    def _append_request_level_rows(
        self, output_dir: str, rows: list[dict[str, Any]]
    ) -> None:
        path = os.path.join(output_dir, "request_level_metrics.jsonl")
        with open(path, "a") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")

    def _build_raw_timing_row(
        self,
        request_id: int,
        session_id: int,
        channel_metrics: dict[str, Any],
        timing: RequestTiming,
    ) -> dict[str, Any]:
        raw_chunks = (
            channel_metrics.get(AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value) or []
        )
        audio_chunks = [
            [round(float(entry[0]), 1), int(entry[1])] for entry in raw_chunks
        ]
        audio_chunks.sort(key=lambda chunk: chunk[0])
        return {
            "request_id": request_id,
            "session_id": session_id,
            "sample_rate": timing.sample_rate,
            "text_deltas": [
                [round(offset, 1), n_chars] for offset, n_chars in timing.text_deltas
            ],
            "audio_chunks": audio_chunks,
            "commit_ms": (
                round(timing.commit_ms, 1) if timing.commit_ms is not None else None
            ),
            "audio_done_ms": (
                round(timing.audio_done_ms, 1)
                if timing.audio_done_ms is not None
                else None
            ),
            "response_done_ms": (
                round(timing.response_done_ms, 1)
                if timing.response_done_ms is not None
                else None
            ),
        }

    def _save_raw_timing(self, output_dir: str) -> None:
        if not self.channel_config.persist_raw_timing or not self._raw_timing_rows:
            return
        path = os.path.join(output_dir, "audio_raw_timing.jsonl")
        with open(path, "w") as file:
            for row in self._raw_timing_rows:
                file.write(json.dumps(row) + "\n")
        self._raw_timing_rows_streamed = len(self._raw_timing_rows)

    def _flush_raw_timing_rows(self, output_dir: str) -> None:
        if not self.channel_config.persist_raw_timing:
            return
        new_rows = self._raw_timing_rows[self._raw_timing_rows_streamed :]
        if not new_rows:
            return
        path = os.path.join(output_dir, "audio_raw_timing.jsonl")
        with open(path, "a") as file:
            for row in new_rows:
                file.write(json.dumps(row) + "\n")
        self._raw_timing_rows_streamed = len(self._raw_timing_rows)

    def _cdf_summaries(self) -> dict[str, CDFSketch]:
        summaries = dict(self.summaries)
        if self._stt_request_count > 0:
            summaries.update(
                {
                    name: sketch
                    for name, sketch in self.asr_latency_summaries.items()
                    if len(sketch) > 0
                }
            )
        return summaries

    def _save_cdf_csvs(self, output_dir: str) -> None:
        for metric_name, cdf_sketch in self._cdf_summaries().items():
            cdf_sketch._to_df().to_csv(
                os.path.join(output_dir, f"audio_{metric_name}.csv"),
                index=False,
            )

    def _plot_cdfs(self, output_dir: str) -> None:
        for metric_name, cdf_sketch in self._cdf_summaries().items():
            try:
                cdf_sketch.plot_cdf(output_dir, f"audio_{metric_name}")
            except Exception as error:
                logger.warning("Failed to plot CDF for %s: %s", metric_name, error)

    def _plot_stall_free_vs_startup_policy(self, output_dir: str) -> None:
        if self._interactive_request_count <= 0:
            return
        try:
            import pandas as pd

            import rekha as rk

            rows: list[dict[str, Any]] = []
            for delay_ms in sorted(self.channel_config.startup_delay_ms_values):
                rows.append(
                    {
                        "startup_budget_ms": delay_ms,
                        "stall_free_fraction": (
                            self._fixed_delay_stall_free_counts.get(delay_ms, 0)
                            / self._interactive_request_count
                        ),
                        "policy": "Fixed delay",
                    }
                )
            for target_ms in sorted(self.channel_config.startup_buffer_ms_values):
                eligible = self._buffer_target_eligible_counts.get(target_ms, 0)
                if eligible == 0:
                    continue
                rows.append(
                    {
                        "startup_budget_ms": target_ms,
                        "stall_free_fraction": (
                            self._buffer_target_stall_free_counts.get(target_ms, 0)
                            / eligible
                        ),
                        "policy": "Buffered audio target",
                    }
                )
            dataframe = pd.DataFrame(rows)
            figure = rk.line(
                dataframe,
                x="startup_budget_ms",
                y="stall_free_fraction",
                color="policy",
                markers=True,
                labels={
                    "startup_budget_ms": "Startup budget (ms)",
                    "stall_free_fraction": "Stall-free fraction",
                    "policy": "Playback policy",
                },
            )
            figure.save(
                os.path.join(output_dir, "audio_stall_free_vs_startup_policy.png"),
                transparent=False,
            )
        except Exception as error:
            logger.warning(
                "Failed to plot stall-free fraction by startup policy: %s", error
            )
