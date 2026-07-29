"""Post-run audio verification with composable verifiers."""

from __future__ import annotations

import json
import math
import statistics
import string
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from veeksha.config.verification import AudioVerificationConfig
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.logger import init_logger

logger = init_logger(__name__)


TranscribeFn = Callable[[Path], str]
_utmos_lock = threading.Lock()
_utmos_jit_model: Any | None = None
_utmos_jit_key: tuple[str, str, str] | None = None
_utmos_jit_load_failed_keys: set[tuple[str, str, str]] = set()


class AudioVerificationError(RuntimeError):
    """Raised when strict audio verification fails."""


@dataclass(frozen=True)
class AudioVerifierContext:
    request_id: int
    reference_text: str
    audio_path: Path
    has_audio: bool


class AudioOutputVerifier(ABC):
    """Base class for a verifier that evaluates persisted audio."""

    name: str

    @abstractmethod
    def verify(self, context: AudioVerifierContext) -> dict[str, Any]:
        """Return row fields produced by this verifier."""


class WERVerifier(AudioOutputVerifier):
    name = "wer"

    def __init__(self, config: AudioVerificationConfig, transcribe_audio: TranscribeFn):
        self.config = config
        self.transcribe_audio = transcribe_audio

    def verify(self, context: AudioVerifierContext) -> dict[str, Any]:
        if not context.reference_text:
            return {
                "wer_error": f"Missing {AudioMetricKey.INPUT_TEXT.value} in request-level metrics"
            }
        if not context.has_audio:
            return {"wer_error": f"Missing audio file: {context.audio_path}"}
        try:
            transcript = self.transcribe_audio(context.audio_path)
            wer = compute_wer(context.reference_text, transcript)
            return {
                "transcript": transcript,
                "wer": wer,
                "passed": wer <= self.config.wer.threshold,
            }
        except Exception as exc:
            return {"wer_error": f"Transcription failed: {exc}"}


class UTMOSVerifier(AudioOutputVerifier):
    name = "utmos"

    def __init__(self, config: AudioVerificationConfig):
        self.config = config

    def verify(self, context: AudioVerifierContext) -> dict[str, Any]:
        if not context.has_audio:
            return {"utmos_error": f"Missing audio file: {context.audio_path}"}
        try:
            utmos = _utmos_predict_audio_path(context.audio_path, self.config)
            if utmos is None:
                return {"utmos_error": "UTMOS unavailable or returned no finite score"}
            return {"utmos": utmos}
        except Exception as exc:
            return {"utmos_error": f"UTMOS failed: {exc}"}


class LocalWhisperTranscriber:
    """In-process faster-whisper transcriber for WER verification."""

    def __init__(self, config: AudioVerificationConfig):
        self.config = config
        whisper_config = config.wer.whisper
        try:
            whisper_model_class = import_module("faster_whisper").WhisperModel
        except ModuleNotFoundError as exc:
            raise AudioVerificationError(
                "WER verification requires faster-whisper. Install the audio-verification extra."
            ) from exc

        self.model = whisper_model_class(
            whisper_config.model,
            device=whisper_config.device,
            compute_type=whisper_config.compute_type,
        )

    def transcribe(self, audio_path: Path) -> str:
        whisper_config = self.config.wer.whisper
        segments, _ = self.model.transcribe(
            str(audio_path),
            language=whisper_config.language,
            task="transcribe",
            beam_size=whisper_config.beam_size,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


@dataclass
class AudioVerificationRow:
    request_id: int
    reference_text: str
    transcript: str
    wer: Optional[float]
    passed: Optional[bool]
    audio_path: str
    utmos: Optional[float] = None
    error: Optional[str] = None
    wer_error: Optional[str] = None
    utmos_error: Optional[str] = None


@dataclass
class AudioVerificationSummary:
    total_requests: int
    evaluated_requests: int
    skipped_requests: int
    max_requests: int
    transcribed_requests: int
    passed_requests: int
    failed_requests: int
    error_requests: int
    wer_avg: Optional[float]
    wer_p50: Optional[float]
    wer_p90: Optional[float]
    wer_p99: Optional[float]
    wer_max: Optional[float]
    wer_threshold: float
    fail_on_threshold: bool
    wer_enabled: bool
    utmos_enabled: bool
    utmos_evaluated: int
    utmos_mean: Optional[float]
    utmos_median: Optional[float]
    utmos_failed: int
    errors: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_text(text: str) -> str:
    """Normalize English text with the exact Seed-TTS-Eval WER protocol.

    Strip all string.punctuation except the apostrophe, collapse double
    spaces in a single replace pass (seed-exact quirk: "a   b" -> "a  b"),
    then lowercase. No number normalization: digits vs words count as errors.
    """
    normalized = text
    for char in string.punctuation:
        if char == "'":
            continue
        normalized = normalized.replace(char, "")
    normalized = normalized.replace("  ", " ")
    return normalized.lower()


def _jiwer_wer(reference: str, hypothesis: str) -> float:
    import jiwer

    compute_measures = getattr(jiwer, "compute_measures", None)
    if compute_measures is not None:
        return float(compute_measures(reference, hypothesis)["wer"])
    return float(jiwer.process_words(reference, hypothesis).wer)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute English WER using Seed-TTS normalization."""
    reference_n = normalize_text(reference)
    hypothesis_n = normalize_text(hypothesis)
    reference_words = reference_n.split()
    hypothesis_words = hypothesis_n.split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    try:
        return _jiwer_wer(reference_n, hypothesis_n)
    except ImportError:
        return _edit_distance(reference_words, hypothesis_words) / len(reference_words)


def _audio_path_to_f32_16k(path: Path) -> np.ndarray:
    import scipy.signal
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if data.size == 0:
        return np.zeros(0, dtype=np.float32)

    mono = np.mean(data, axis=1).astype(np.float32)
    if int(sample_rate) == 16000:
        return mono

    target_len = max(1, int(len(mono) * 16000 / int(sample_rate)))
    return np.asarray(scipy.signal.resample(mono, target_len), dtype=np.float32)


def _resolve_utmos_device(device: str) -> str:
    torch: Any = import_module("torch")

    device = device.strip()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise AudioVerificationError(
            f"UTMOS requested {device}, but CUDA is unavailable."
        )
    return device


def load_utmos_jit_model(hf_repo: str, jit_file: str, device: str) -> Any | None:
    """Load (and cache) the UTMOS TorchScript model, or None if unavailable."""
    global _utmos_jit_key, _utmos_jit_model

    key = (hf_repo, jit_file, device)
    with _utmos_lock:
        if key in _utmos_jit_load_failed_keys:
            return None
        if _utmos_jit_model is not None and _utmos_jit_key == key:
            return _utmos_jit_model

        try:
            torch: Any = import_module("torch")
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=hf_repo,
                filename=jit_file,
                repo_type="model",
            )
            model = torch.jit.load(path, map_location=_resolve_utmos_device(device))
            model.eval()
        except Exception as exc:
            logger.warning(
                "UTMOS JIT unavailable; install torch, scipy, soundfile, and "
                "huggingface_hub, then check GPU/model access: %s",
                exc,
            )
            _utmos_jit_load_failed_keys.add(key)
            return None

        _utmos_jit_model = model
        _utmos_jit_key = key
        return _utmos_jit_model


def predict_utmos_f32_16k(
    wav_f32: np.ndarray, hf_repo: str, jit_file: str, device: str
) -> float | None:
    """Score 16 kHz mono float32 audio with UTMOS, or None if unavailable."""
    torch: Any = import_module("torch")

    if len(wav_f32) == 0:
        return None

    model = load_utmos_jit_model(hf_repo, jit_file, device)
    if model is None:
        return None

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        try:
            model_device = next(model.buffers()).device
        except StopIteration:
            model_device = torch.device(device)

    wav = np.ascontiguousarray(wav_f32, dtype=np.float32)
    model_input = (
        torch.from_numpy(wav).unsqueeze(0).to(device=model_device, dtype=torch.float32)
    )
    with torch.no_grad():
        output = model(model_input)
    value = float(output.reshape(-1)[0].item())
    if not math.isfinite(value):
        return None
    return value


def _utmos_predict_audio_path(
    audio_path: Path, config: AudioVerificationConfig
) -> float | None:
    wav_16k = _audio_path_to_f32_16k(audio_path)
    return predict_utmos_f32_16k(
        wav_16k, config.utmos.hf_repo, config.utmos.jit_file, config.utmos.device
    )


def build_audio_verifiers(
    config: AudioVerificationConfig,
    transcribe_audio: Optional[TranscribeFn] = None,
) -> list[AudioOutputVerifier]:
    """Build the configured audio output verifiers."""
    verifiers: list[AudioOutputVerifier] = []
    if config.wer.enabled:
        if transcribe_audio is None:
            raise AudioVerificationError(
                "WER verification requires a transcription function"
            )
        verifiers.append(WERVerifier(config, transcribe_audio))
    if config.utmos.enabled:
        verifiers.append(UTMOSVerifier(config))
    return verifiers


def verify_audio_outputs(
    output_dir: str | Path,
    config: AudioVerificationConfig,
    transcribe_audio: Optional[TranscribeFn] = None,
) -> AudioVerificationSummary:
    """Verify saved audio files and persist JSON artifacts."""
    verifiers = build_audio_verifiers(config, transcribe_audio)
    output_path = Path(output_dir)
    verification_dir = output_path / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)

    request_metrics_path = output_path / "metrics" / "request_level_metrics.jsonl"
    audio_dir = output_path / "audio_files"
    rows: list[AudioVerificationRow] = []
    errors: list[str] = []
    total_requests = 0
    skipped_requests = 0

    if not request_metrics_path.exists():
        errors.append(f"Missing request metrics file: {request_metrics_path}")
    else:
        metric_rows = list(_load_jsonl(request_metrics_path))
        total_requests = len(metric_rows)
        if config.max_requests > 0:
            skipped_requests = max(total_requests - config.max_requests, 0)
            metric_rows = metric_rows[: config.max_requests]
        if skipped_requests:
            logger.info(
                "Audio verification limited to first %d of %d request rows",
                config.max_requests,
                total_requests,
            )

        for metric_row in metric_rows:
            request_id = metric_row.get("request_id")
            if request_id is None:
                errors.append("Skipping request row without request_id")
                continue

            audio_path = audio_dir / f"request_{request_id}.wav"
            context = AudioVerifierContext(
                request_id=int(request_id),
                reference_text=str(
                    metric_row.get(AudioMetricKey.INPUT_TEXT.value) or ""
                ),
                audio_path=audio_path,
                has_audio=audio_path.exists(),
            )
            row_data: dict[str, Any] = {
                "transcript": "",
                "wer": None,
                "passed": None,
                "utmos": None,
                "wer_error": None,
                "utmos_error": None,
            }
            for verifier in verifiers:
                row_data.update(verifier.verify(context))

            row_error = _combine_errors(row_data["wer_error"], row_data["utmos_error"])
            rows.append(
                AudioVerificationRow(
                    request_id=context.request_id,
                    reference_text=context.reference_text,
                    transcript=row_data["transcript"],
                    wer=row_data["wer"],
                    passed=row_data["passed"],
                    audio_path=str(context.audio_path),
                    utmos=row_data["utmos"],
                    error=row_error,
                    wer_error=row_data["wer_error"],
                    utmos_error=row_data["utmos_error"],
                )
            )

    _save_rows(verification_dir / "audio_verification.jsonl", rows)
    summary = _build_summary(rows, config, errors, total_requests, skipped_requests)
    _save_summary(verification_dir / "audio_summary.json", summary)

    if summary.failed_requests:
        logger.warning(
            "Audio verification found %d requests above WER threshold %.4f",
            summary.failed_requests,
            config.wer.threshold,
        )
    if summary.error_requests or summary.errors:
        logger.warning(
            "Audio verification completed with %d request errors and %d run errors",
            summary.error_requests,
            len(summary.errors),
        )

    if config.fail_on_threshold:
        failures = []
        if summary.failed_requests:
            failures.append(
                f"{summary.failed_requests} audio requests exceeded WER threshold "
                f"{config.wer.threshold}"
            )
        if summary.error_requests:
            failures.append(
                f"{summary.error_requests} audio requests could not be verified"
            )
        if summary.errors:
            failures.append(f"{len(summary.errors)} run-level verification errors")
        if failures:
            raise AudioVerificationError("; ".join(failures))

    return summary


def run_audio_verification(
    config: AudioVerificationConfig,
    output_dir: str | Path,
) -> AudioVerificationSummary:
    """Run configured post-run audio verification metrics."""
    verification_dir = Path(output_dir) / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)

    transcriber = LocalWhisperTranscriber(config) if config.wer.enabled else None
    return verify_audio_outputs(
        output_dir=output_dir,
        config=config,
        transcribe_audio=transcriber.transcribe if transcriber is not None else None,
    )


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _save_rows(path: Path, rows: list[AudioVerificationRow]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")


def _save_summary(path: Path, summary: AudioVerificationSummary) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)


def _build_summary(
    rows: list[AudioVerificationRow],
    config: AudioVerificationConfig,
    errors: list[str],
    total_requests: int,
    skipped_requests: int,
) -> AudioVerificationSummary:
    wers = [row.wer for row in rows if row.wer is not None]
    utmos_values = [row.utmos for row in rows if row.utmos is not None]
    passed_requests = sum(1 for row in rows if row.passed is True)
    failed_requests = sum(1 for row in rows if row.passed is False)
    error_requests = sum(1 for row in rows if row.error is not None)
    utmos_failed = sum(1 for row in rows if row.utmos_error is not None)
    return AudioVerificationSummary(
        total_requests=total_requests,
        evaluated_requests=len(rows),
        skipped_requests=skipped_requests,
        max_requests=config.max_requests,
        transcribed_requests=len(wers),
        passed_requests=passed_requests,
        failed_requests=failed_requests,
        error_requests=error_requests,
        wer_avg=(sum(wers) / len(wers)) if wers else None,
        wer_p50=_percentile(wers, 0.50),
        wer_p90=_percentile(wers, 0.90),
        wer_p99=_percentile(wers, 0.99),
        wer_max=max(wers) if wers else None,
        wer_threshold=config.wer.threshold,
        fail_on_threshold=config.fail_on_threshold,
        wer_enabled=config.wer.enabled,
        utmos_enabled=config.utmos.enabled,
        utmos_evaluated=len(utmos_values),
        utmos_mean=statistics.fmean(utmos_values) if utmos_values else None,
        utmos_median=statistics.median(utmos_values) if utmos_values else None,
        utmos_failed=utmos_failed,
        errors=errors,
    )


def _combine_errors(*errors: Optional[str]) -> Optional[str]:
    parts = [error for error in errors if error]
    return "; ".join(parts) if parts else None


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, len(ordered) - 1)
    fraction = rank - lower_idx
    return ordered[lower_idx] + (ordered[upper_idx] - ordered[lower_idx]) * fraction


def _edit_distance(reference_words: list[str], hypothesis_words: list[str]) -> int:
    previous = list(range(len(hypothesis_words) + 1))
    for i, reference_word in enumerate(reference_words, start=1):
        current = [i]
        for j, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution_cost = 0 if reference_word == hypothesis_word else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]
