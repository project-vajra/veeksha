"""Generated audio quality evaluator."""

from __future__ import annotations

import json
import struct
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Set

from veeksha.config.evaluator import AudioQualityEvaluatorConfig
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.seeding import SeedManager
from veeksha.evaluator.accuracy.base import BaseAccuracyEvaluator
from veeksha.evaluator.base import EvaluationResult
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

logger = init_logger(__name__)

DEFAULT_AUDIO_SAMPLE_RATE = 24000


@dataclass
class AudioQualityRequestRow:
    request_id: int
    session_id: int
    input_text: str
    audio_present: bool
    error: Optional[str] = None


def _make_wav_header(
    data_size: int, sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
) -> bytes:
    """Build a 44-byte mono 16-bit PCM WAV header."""
    byte_rate = sample_rate * 1 * 16 // 8
    block_align = 1 * 16 // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )


class AudioQualityEvaluator(BaseAccuracyEvaluator):
    """Accuracy-family evaluator for generated audio quality.

    It persists an audio-quality-scoped copy of generated audio and reference
    text, then runs configured audio verifiers such as WER or UTMOS.
    """

    def __init__(
        self,
        config: AudioQualityEvaluatorConfig,
        seed_manager: Optional[SeedManager] = None,
        output_dir: Optional[str] = None,
        benchmark_start_time: float = 0.0,
    ):
        super().__init__(
            config=config,
            seed_manager=seed_manager,
            output_dir=output_dir,
            benchmark_start_time=benchmark_start_time,
        )
        self.config: AudioQualityEvaluatorConfig = config
        self._lock = threading.Lock()
        self._registered_request_ids: Set[int] = set()
        self._included_request_ids: Optional[Set[int]] = None
        self._reference_text_by_request_id: Dict[int, str] = {}
        self._rows: list[AudioQualityRequestRow] = []
        self._audio_buffers: Dict[int, bytes] = {}
        self._audio_metadata: Dict[int, Dict[str, Any]] = {}
        self.num_requests = 0
        self.num_completed_requests = 0
        self.num_errored_requests = 0
        self._verification_summary: Optional[dict[str, Any]] = None

    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        channels: Dict[ChannelModality, Any],
        requested_output: Any = None,
    ) -> None:
        if not self.should_evaluate_channel(ChannelModality.AUDIO):
            return
        with self._lock:
            self.num_requests += 1
            self._registered_request_ids.add(request_id)
            text_content = channels.get(ChannelModality.TEXT)
            if isinstance(text_content, TextChannelRequestContent):
                self._reference_text_by_request_id[request_id] = text_content.input_text

    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
        error: Optional[Exception] = None,
    ) -> None:
        if not self.should_evaluate_channel(ChannelModality.AUDIO):
            return
        with self._lock:
            if (
                self._included_request_ids is not None
                and request_id not in self._included_request_ids
            ):
                return

            input_text = self._reference_text_by_request_id.get(request_id, "")
            if error is not None or not getattr(response, "success", True):
                self.num_errored_requests += 1
                self._rows.append(
                    AudioQualityRequestRow(
                        request_id=request_id,
                        session_id=session_id,
                        input_text=input_text,
                        audio_present=False,
                        error=str(
                            error or getattr(response, "error_msg", "request failed")
                        ),
                    )
                )
                return

            audio_channel = getattr(response, "channels", {}).get(ChannelModality.AUDIO)
            audio_content = (
                getattr(audio_channel, "content", None) if audio_channel else None
            )
            if not isinstance(audio_content, bytes) or not audio_content:
                self.num_errored_requests += 1
                self._rows.append(
                    AudioQualityRequestRow(
                        request_id=request_id,
                        session_id=session_id,
                        input_text=input_text,
                        audio_present=False,
                        error="missing audio channel content",
                    )
                )
                return

            metrics = getattr(audio_channel, "metrics", {}) or {}
            self.num_completed_requests += 1
            self._rows.append(
                AudioQualityRequestRow(
                    request_id=request_id,
                    session_id=session_id,
                    input_text=input_text,
                    audio_present=True,
                )
            )
            if self._should_save_audio_files():
                self._audio_buffers[request_id] = audio_content
                self._audio_metadata[request_id] = {
                    "raw_pcm": bool(metrics.get("raw_pcm", False)),
                    "sample_rate": int(
                        metrics.get("sample_rate", DEFAULT_AUDIO_SAMPLE_RATE)
                    ),
                }

    def record_session_completed(
        self,
        session_id: int,
        completed_at: float,
        success: bool,
    ) -> None:
        return

    def finalize(self) -> EvaluationResult:
        with self._lock:
            metrics = {
                "num_requests": self.num_requests,
                "num_completed_requests": self.num_completed_requests,
                "num_errored_requests": self.num_errored_requests,
                "verification_enabled": self.config.verification.is_enabled(),
            }
            if self._verification_summary is not None:
                metrics["verification"] = self._verification_summary
            return EvaluationResult(
                evaluator_type="audio_quality",
                channel=ChannelModality.AUDIO,
                metrics=metrics,
            )

    def save(self, output_dir: str) -> None:
        audio_quality_dir = Path(output_dir).parent / "audio_quality"
        metrics_dir = audio_quality_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self._save_request_rows(metrics_dir / "request_level_metrics.jsonl")
        if self._should_save_audio_files():
            self._save_audio_files(audio_quality_dir / "audio_files")
        self._maybe_run_verification(audio_quality_dir)

    def get_completed_request_count(self) -> int:
        with self._lock:
            return self.num_completed_requests

    def get_session_counts(self) -> tuple[int, int, int]:
        with self._lock:
            in_progress = max(
                0,
                len(self._registered_request_ids)
                - self.num_completed_requests
                - self.num_errored_requests,
            )
            return self.num_completed_requests, self.num_errored_requests, in_progress

    def set_included_requests(self, request_ids: Set[int]) -> None:
        with self._lock:
            self._included_request_ids = set(request_ids)

    def get_registered_request_ids(self) -> Set[int]:
        with self._lock:
            return set(self._registered_request_ids)

    def _should_save_audio_files(self) -> bool:
        return bool(self.config.save_audio_files)

    def _save_request_rows(self, path: Path) -> None:
        with self._lock:
            rows = list(self._rows)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(asdict(row)) + "\n")

    def _save_audio_files(self, audio_dir: Path) -> None:
        with self._lock:
            buffers = dict(self._audio_buffers)
            metadata = dict(self._audio_metadata)
        if not buffers:
            return
        audio_dir.mkdir(parents=True, exist_ok=True)
        for request_id, audio_data in buffers.items():
            meta = metadata.get(request_id, {})
            raw_pcm = bool(meta.get("raw_pcm", False))
            sample_rate = int(meta.get("sample_rate", DEFAULT_AUDIO_SAMPLE_RATE))
            wav_path = audio_dir / f"request_{request_id}.wav"
            with wav_path.open("wb") as f:
                if raw_pcm:
                    f.write(_make_wav_header(len(audio_data), sample_rate))
                f.write(audio_data)
        logger.info("Saved %d audio quality files to %s", len(buffers), audio_dir)

    def _maybe_run_verification(self, audio_quality_dir: Path) -> None:
        verification_config = self.config.verification
        if not verification_config.is_enabled():
            return
        if not self._should_save_audio_files():
            logger.warning(
                "Audio quality verification is enabled but audio saving is disabled; "
                "skipping because no audio artifacts were saved."
            )
            return

        from veeksha.verification.audio import (
            AudioVerificationError,
            run_audio_verification,
        )

        try:
            summary = run_audio_verification(
                config=verification_config,
                output_dir=audio_quality_dir,
            )
            summary_dict = summary.to_dict()
            with self._lock:
                self._verification_summary = summary_dict
            logger.info(
                "Audio quality verification complete: %d transcribed, %d above threshold, "
                "%d UTMOS scored, %d errors",
                summary.transcribed_requests,
                summary.failed_requests,
                summary.utmos_evaluated,
                summary.error_requests + len(summary.errors),
            )
        except AudioVerificationError:
            if verification_config.fail_on_threshold:
                raise
            logger.exception("Audio quality verification failed; continuing")
        except Exception:
            if verification_config.fail_on_threshold:
                raise
            logger.exception(
                "Audio quality verification failed; continuing because fail_on_threshold=False"
            )
