"""Performance evaluator for TTS audio generation metrics."""

import json
import os
import re
import string
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from veeksha.config.evaluator import (
    AudioChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.evaluator.base import EvaluationResult
from veeksha.evaluator.cdf_sketch import CDFSketch
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

logger = init_logger(__name__)

SAMPLE_RATE_DEFAULT = 24000

# ---- Text normalization for WER ----

_ONES = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
}

_ORDINALS = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth",
}

_CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "it's": "it is", "i'm": "i am",
    "i've": "i have", "i'll": "i will", "i'd": "i would",
    "he's": "he is", "she's": "she is", "they're": "they are",
    "we're": "we are", "you're": "you are", "that's": "that is",
    "there's": "there is", "what's": "what is",
}

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_transcript(text: str) -> str:
    """Normalize transcript for fair WER comparison.

    Lowercase → expand contractions → expand ordinals →
    expand numbers (0-20) → remove punctuation → collapse whitespace.
    """
    text = text.lower().strip()
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    for ordinal, word in _ORDINALS.items():
        text = re.sub(r"\b" + re.escape(ordinal) + r"\b", word, text)
    for digit, word in _ONES.items():
        text = re.sub(r"\b" + re.escape(digit) + r"\b", word, text)
    text = text.translate(_PUNCTUATION_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate via Levenshtein on word sequences.

    Returns WER as a percentage (0-100+).
    """
    ref_words = _normalize_transcript(reference).split()
    hyp_words = _normalize_transcript(hypothesis).split()

    if not ref_words:
        return 0.0 if not hyp_words else 100.0

    # Levenshtein DP on word lists
    n, m = len(ref_words), len(hyp_words)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr

    return (prev[m] / n) * 100


def _make_wav_header(data_size: int, sample_rate: int = SAMPLE_RATE_DEFAULT) -> bytes:
    """Build a 44-byte WAV header for raw PCM data."""
    byte_rate = sample_rate * 1 * 16 // 8
    block_align = 1 * 16 // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        1,  # mono
        sample_rate,
        byte_rate,
        block_align,
        16,  # bits per sample
        b"data",
        data_size,
    )


@dataclass
class AudioRequestMetrics:
    """Metrics for a single audio (TTS) request."""

    request_id: int
    session_id: int
    request_dispatched_at: float
    client_completed_at: float
    ttft: float
    e2e: float
    generated_audio_duration: float
    rtf: float
    chunk_count: int
    pcm_byte_count: int
    input_chars: int = 0
    input_tokens: int = 0
    ttfa: Optional[float] = None
    tpot: Optional[float] = None
    final_latency: Optional[float] = None
    first_partial: Optional[float] = None
    wer: Optional[float] = None
    transcript: Optional[str] = None
    expected_transcript: Optional[str] = None
    input_text: str = ""
    session_total_requests: Optional[int] = None


class AudioPerformanceEvaluator:
    """Performance evaluator for audio (TTS / STT).

    Tracks per-request TTFT, total latency, audio duration, RTF (real-time factor),
    and chunk count. Computes p50/p90/p99 aggregates via CDFSketch.
    """

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        channel_config: Optional[AudioChannelPerformanceConfig] = None,
        benchmark_start_time: float = 0.0,
    ):
        self.config = config
        self.channel_config = channel_config or AudioChannelPerformanceConfig()
        self.benchmark_start_time = benchmark_start_time
        self.lock = threading.Lock()

        # CDF sketches for aggregate metrics
        self.summaries: Dict[str, CDFSketch] = {
            "ttft": CDFSketch("ttft", unit="ms"),
            "ttfa": CDFSketch("ttfa", unit="ms"),
            "tpot": CDFSketch("tpot", unit="ms"),
            "final_latency": CDFSketch("final_latency", unit="ms"),
            "first_partial": CDFSketch("first_partial", unit="ms"),
            "e2e": CDFSketch("e2e", unit="ms"),
            "generated_audio_duration": CDFSketch(
                "generated_audio_duration", unit="ms"
            ),
            "rtf": CDFSketch("rtf"),
            "chunk_count": CDFSketch("chunk_count"),
            "input_tokens": CDFSketch("input_tokens", unit="tokens"),
            "wer": CDFSketch("wer", unit="%"),
            "session_size": CDFSketch("session_size"),
            "session_duration": CDFSketch("session_duration", unit="ms"),
        }

        # Request dispatch tracking (like text.py _pending_requests)
        self._pending_requests: Dict[int, Dict[str, Any]] = {}

        # Request-level storage for JSONL output
        self._completed_metrics: List[AudioRequestMetrics] = []
        self._lifecycle_timestamps: List[Dict[str, Optional[float]]] = []
        self._request_rows_streamed: int = 0
        self._request_time_reference: float = self.benchmark_start_time

        # Audio bytes storage for optional WAV saving
        self._audio_buffers: Dict[int, bytes] = {}
        self._audio_metadata: Dict[int, Dict[str, Any]] = {}

        # Running totals for aggregate throughput
        self._total_input_chars: int = 0
        self._total_generated_audio_duration_ms: float = 0.0
        self._first_dispatch_at: Optional[float] = None
        self._last_completion_at: Optional[float] = None

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
        """Record that an audio request completed."""
        with self.lock:
            dispatch_info = self._pending_requests.pop(request_id, None)
            if dispatch_info:
                dispatched_at = dispatch_info["dispatched_at"]
            else:
                dispatched_at = getattr(
                    response,
                    "scheduler_dispatched_at",
                    self._request_time_reference,
                )

            channel_response = None
            if hasattr(response, "channels"):
                channel_response = response.channels.get(ChannelModality.AUDIO)

            if channel_response is None:
                logger.debug("Request %d has no AUDIO channel response", request_id)
                return

            cm = channel_response.metrics or {}
            ttft = cm.get("ttft", 0.0)
            ttfa_value: Optional[float] = cm.get("ttfa")
            tpot_value: Optional[float] = cm.get("tpot")
            final_latency_value: Optional[float] = cm.get("final_latency")
            first_partial_value: Optional[float] = cm.get("first_partial")
            e2e = cm.get("e2e", 0.0)
            generated_audio_duration = cm.get("generated_audio_duration", 0.0)
            rtf = cm.get("rtf", 0.0)
            chunk_count = cm.get("chunk_count", 0)
            pcm_byte_count = cm.get("pcm_byte_count", 0)
            input_chars = cm.get("input_chars", 0)
            input_tokens = cm.get("input_tokens", 0)
            input_text = cm.get("input_text", "")

            session_total_requests = getattr(response, "session_total_requests", None)

            # WER computation
            transcript = cm.get("transcript")
            expected_transcript = cm.get("expected_transcript")
            wer_value: Optional[float] = None
            if transcript is not None and expected_transcript is not None:
                wer_value = _compute_wer(expected_transcript, transcript)

            metrics = AudioRequestMetrics(
                request_id=request_id,
                session_id=session_id,
                request_dispatched_at=dispatched_at,
                client_completed_at=completed_at,
                ttft=ttft,
                e2e=e2e,
                generated_audio_duration=generated_audio_duration,
                rtf=rtf,
                chunk_count=chunk_count,
                pcm_byte_count=pcm_byte_count,
                input_chars=input_chars,
                input_tokens=input_tokens,
                ttfa=ttfa_value,
                tpot=tpot_value,
                final_latency=final_latency_value,
                first_partial=first_partial_value,
                wer=wer_value,
                transcript=transcript,
                expected_transcript=expected_transcript,
                input_text=input_text,
                session_total_requests=session_total_requests,
            )

            self._completed_metrics.append(metrics)

            # Update aggregate throughput accumulators
            self._total_input_chars += input_chars
            self._total_generated_audio_duration_ms += generated_audio_duration
            if self._first_dispatch_at is None or dispatched_at < self._first_dispatch_at:
                self._first_dispatch_at = dispatched_at
            if self._last_completion_at is None or completed_at > self._last_completion_at:
                self._last_completion_at = completed_at

            # Store lifecycle timestamps (matching text.py pattern)
            def normalize_ts(ts: Optional[float]) -> Optional[float]:
                if ts is None:
                    return None
                return round(max(0.0, ts - self._request_time_reference), 5)

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

            # Update CDF sketches
            self.summaries["ttft"].put(ttft)
            if ttfa_value is not None:
                self.summaries["ttfa"].put(ttfa_value)
            if tpot_value is not None:
                self.summaries["tpot"].put(tpot_value)
            if final_latency_value is not None:
                self.summaries["final_latency"].put(final_latency_value)
            if first_partial_value is not None:
                self.summaries["first_partial"].put(first_partial_value)
            self.summaries["e2e"].put(e2e)
            self.summaries["generated_audio_duration"].put(generated_audio_duration)
            self.summaries["rtf"].put(rtf)
            self.summaries["chunk_count"].put(chunk_count)
            self.summaries["input_tokens"].put(input_tokens)
            if wer_value is not None:
                self.summaries["wer"].put(wer_value)

            # Store audio buffer for optional WAV saving
            if self.channel_config.save_audio_files:
                audio_content = channel_response.content
                if audio_content and isinstance(audio_content, bytes):
                    self._audio_buffers[request_id] = audio_content
                    self._audio_metadata[request_id] = {
                        "raw_pcm": cm.get("raw_pcm", False),
                        "sample_rate": cm.get("sample_rate", SAMPLE_RATE_DEFAULT),
                    }

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

            if first_dispatch_at is not None and last_completion_at is not None:
                duration = max(0.0, last_completion_at - first_dispatch_at)
                self.summaries["session_duration"].put(duration)

    def get_summary(self) -> Dict[str, Optional[float]]:
        """Get summary metrics from all CDF sketches."""
        perf_summary: Dict[str, Optional[float]] = {}
        for cdf_sketch in self.summaries.values():
            perf_summary.update(cdf_sketch.get_summary())

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
        return perf_summary

    def finalize(self) -> EvaluationResult:
        """Finalize evaluation and return results."""
        with self.lock:
            return EvaluationResult(
                evaluator_type="audio_performance",
                channel=ChannelModality.AUDIO,
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
            self._plot_cdfs(output_dir)
            if self.channel_config.save_audio_files:
                # Save audio files to parent dir (not inside metrics/)
                parent_dir = os.path.dirname(output_dir)
                self._save_audio_files(parent_dir)

    def flush_streaming_outputs(self, output_dir: str) -> None:
        """Flush current metrics for streaming."""
        with self.lock:
            rows = self._export_request_rows(self._request_rows_streamed)
            if rows:
                self._append_request_level_rows(output_dir, rows)
                self._request_rows_streamed = len(self._completed_metrics)
            self._save_cdf_csvs(output_dir)

    # ---- Output helpers ----

    def _save_request_level_metrics(self, output_dir: str) -> None:
        """Save request-level audio metrics as JSONL."""
        path = os.path.join(output_dir, "request_level_metrics.jsonl")
        rows = self._export_request_rows(0)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _export_request_rows(self, start_index: int = 0) -> List[Dict[str, Any]]:
        rows = []
        for idx in range(start_index, len(self._completed_metrics)):
            m = self._completed_metrics[idx]
            lifecycle = self._lifecycle_timestamps[idx]
            normalized_dispatched = max(
                0.0, m.request_dispatched_at - self._request_time_reference
            )
            normalized_completed = max(
                0.0, m.client_completed_at - self._request_time_reference
            )
            row_dict = {
                "request_id": m.request_id,
                "session_id": m.session_id,
                "session_total_requests": m.session_total_requests,
                # Lifecycle timestamps
                "scheduler_ready_at": lifecycle["scheduler_ready_at"],
                "scheduler_dispatched_at": round(normalized_dispatched, 5),
                "client_picked_up_at": lifecycle["client_picked_up_at"],
                "client_completed_at": round(normalized_completed, 5),
                "result_processed_at": lifecycle["result_processed_at"],
                "ttft": round(m.ttft, 3),
                "e2e": round(m.e2e, 3),
                "generated_audio_duration": round(m.generated_audio_duration, 3),
                "rtf": round(m.rtf, 5),
                "chunk_count": m.chunk_count,
                "pcm_byte_count": m.pcm_byte_count,
                "input_chars": m.input_chars,
                "input_tokens": m.input_tokens,
                "input_text": m.input_text,
            }
            if m.ttfa is not None:
                row_dict["ttfa"] = round(m.ttfa, 3)
            if m.tpot is not None:
                row_dict["tpot"] = round(m.tpot, 3)
            if m.final_latency is not None:
                row_dict["final_latency"] = round(m.final_latency, 3)
            if m.first_partial is not None:
                row_dict["first_partial"] = round(m.first_partial, 3)
            if m.wer is not None:
                row_dict["wer"] = round(m.wer, 3)
            if m.transcript is not None:
                row_dict["transcript"] = m.transcript
            if m.expected_transcript is not None:
                row_dict["expected_transcript"] = m.expected_transcript
            rows.append(row_dict)
        return rows

    def _append_request_level_rows(
        self, output_dir: str, rows: List[Dict[str, Any]]
    ) -> None:
        path = os.path.join(output_dir, "request_level_metrics.jsonl")
        with open(path, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _save_cdf_csvs(self, output_dir: str) -> None:
        for metric_name, cdf_sketch in self.summaries.items():
            df = cdf_sketch._to_df()
            df.to_csv(
                os.path.join(output_dir, f"audio_{metric_name}.csv"),
                index=False,
            )

    def _plot_cdfs(self, output_dir: str) -> None:
        for metric_name, cdf_sketch in self.summaries.items():
            try:
                cdf_sketch.plot_cdf(output_dir, f"audio_{metric_name}")
            except Exception as e:
                logger.warning("Failed to plot CDF for %s: %s", metric_name, e)

    def _save_audio_files(self, output_dir: str) -> None:
        """Save collected audio buffers as WAV files."""
        if not self._audio_buffers:
            return
        audio_dir = os.path.join(output_dir, "audio_files")
        os.makedirs(audio_dir, exist_ok=True)
        for req_id, audio_data in self._audio_buffers.items():
            meta = self._audio_metadata.get(req_id, {})
            raw_pcm = meta.get("raw_pcm", False)
            sample_rate = meta.get("sample_rate", SAMPLE_RATE_DEFAULT)
            wav_path = os.path.join(audio_dir, f"request_{req_id}.wav")
            with open(wav_path, "wb") as f:
                if raw_pcm:
                    f.write(_make_wav_header(len(audio_data), sample_rate))
                f.write(audio_data)
        logger.info("Saved %d audio files to %s", len(self._audio_buffers), audio_dir)
        self._audio_buffers.clear()
        self._audio_metadata.clear()
