"""Post-run TTS verification with Whisper transcription, WER, and UTMOS."""

from __future__ import annotations

import json
import math
import os
import socket
import statistics
import string
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import requests

from veeksha.config.verification import TTSVerificationConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


TranscribeFn = Callable[[Path], str]
_utmos_lock = threading.Lock()
_utmos_jit_model: Any | None = None
_utmos_jit_key: tuple[str, str, str] | None = None
_utmos_jit_load_failed_keys: set[tuple[str, str, str]] = set()


class TTSVerificationError(RuntimeError):
    """Raised when strict TTS verification fails."""


@dataclass
class TTSVerificationRow:
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
class TTSVerificationSummary:
    total_requests: int
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
    """Normalize English text with the Seed-TTS WER punctuation protocol."""
    normalized = text
    for char in string.punctuation:
        if char == "'":
            continue
        normalized = normalized.replace(char, "")
    return normalized.lower()


def _jiwer_wer(reference: str, hypothesis: str) -> float:
    try:
        from jiwer import compute_measures

        return float(compute_measures(reference, hypothesis)["wer"])
    except ImportError:
        import jiwer

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
    return scipy.signal.resample(mono, target_len).astype(np.float32)


def _utmos_key(config: TTSVerificationConfig) -> tuple[str, str, str]:
    return (config.utmos_hf_repo, config.utmos_jit_file, config.utmos_device)


def _resolve_utmos_device(config: TTSVerificationConfig) -> str:
    import torch

    device = config.utmos_device.strip()
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "UTMOS device %s requested but CUDA is unavailable; using CPU", device
        )
        return "cpu"
    return device


def _ensure_utmos_jit_model(config: TTSVerificationConfig) -> Any | None:
    global _utmos_jit_key, _utmos_jit_model

    key = _utmos_key(config)
    with _utmos_lock:
        if key in _utmos_jit_load_failed_keys:
            return None
        if _utmos_jit_model is not None and _utmos_jit_key == key:
            return _utmos_jit_model

        try:
            import torch
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=config.utmos_hf_repo,
                filename=config.utmos_jit_file,
                repo_type="model",
            )
            target_device = _resolve_utmos_device(config)
            try:
                model = torch.jit.load(path, map_location=target_device)
            except Exception as exc:
                if target_device.startswith("cuda"):
                    logger.warning(
                        "UTMOS JIT load on %s failed (%s), retrying on CPU",
                        target_device,
                        exc,
                    )
                    model = torch.jit.load(path, map_location="cpu")
                else:
                    raise
            model.eval()
        except Exception as exc:
            logger.warning(
                "UTMOS JIT unavailable; install torch, scipy, soundfile, and "
                "huggingface_hub, then check HF access: %s",
                exc,
            )
            _utmos_jit_load_failed_keys.add(key)
            return None

        _utmos_jit_model = model
        _utmos_jit_key = key
        return _utmos_jit_model


def _utmos_predict_f32_16k(
    wav_f32: np.ndarray, config: TTSVerificationConfig
) -> float | None:
    import torch

    if len(wav_f32) == 0:
        return None

    model = _ensure_utmos_jit_model(config)
    if model is None:
        return None

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        try:
            model_device = next(model.buffers()).device
        except StopIteration:
            model_device = torch.device("cpu")

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
    audio_path: Path, config: TTSVerificationConfig
) -> float | None:
    wav_16k = _audio_path_to_f32_16k(audio_path)
    return _utmos_predict_f32_16k(wav_16k, config)


def verify_tts_outputs(
    output_dir: str | Path,
    config: TTSVerificationConfig,
    transcribe_audio: Optional[TranscribeFn] = None,
) -> TTSVerificationSummary:
    """Verify saved TTS audio files and persist JSON artifacts."""
    if config.wer_enabled and transcribe_audio is None:
        raise TTSVerificationError("WER verification requires a transcription function")

    output_path = Path(output_dir)
    verification_dir = output_path / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)

    request_metrics_path = output_path / "metrics" / "request_level_metrics.jsonl"
    audio_dir = output_path / "audio_files"
    rows: list[TTSVerificationRow] = []
    errors: list[str] = []

    if not request_metrics_path.exists():
        errors.append(f"Missing request metrics file: {request_metrics_path}")
    else:
        for metric_row in _load_jsonl(request_metrics_path):
            request_id = metric_row.get("request_id")
            if request_id is None:
                errors.append("Skipping request row without request_id")
                continue

            audio_path = audio_dir / f"request_{request_id}.wav"
            reference_text = str(metric_row.get("input_text") or "")
            has_audio = audio_path.exists()
            transcript = ""
            wer: Optional[float] = None
            passed: Optional[bool] = None
            utmos: Optional[float] = None
            wer_error: Optional[str] = None
            utmos_error: Optional[str] = None

            if config.wer_enabled:
                if not reference_text:
                    wer_error = "Missing input_text in request-level metrics"
                elif not has_audio:
                    wer_error = f"Missing audio file: {audio_path}"
                else:
                    try:
                        assert transcribe_audio is not None
                        transcript = transcribe_audio(audio_path)
                        wer = compute_wer(reference_text, transcript)
                        passed = wer <= config.wer_threshold
                    except Exception as exc:
                        wer_error = f"Transcription failed: {exc}"

            if config.utmos_enabled:
                if not has_audio:
                    utmos_error = f"Missing audio file: {audio_path}"
                else:
                    try:
                        utmos = _utmos_predict_audio_path(audio_path, config)
                        if utmos is None:
                            utmos_error = (
                                "UTMOS unavailable or returned no finite score"
                            )
                    except Exception as exc:
                        utmos_error = f"UTMOS failed: {exc}"

            row_error = _combine_errors(wer_error, utmos_error)
            rows.append(
                TTSVerificationRow(
                    request_id=int(request_id),
                    reference_text=reference_text,
                    transcript=transcript,
                    wer=wer,
                    passed=passed,
                    audio_path=str(audio_path),
                    utmos=utmos,
                    error=row_error,
                    wer_error=wer_error,
                    utmos_error=utmos_error,
                )
            )

    _save_rows(verification_dir / "tts_whisper_verification.jsonl", rows)
    summary = _build_summary(rows, config, errors)
    _save_summary(verification_dir / "tts_whisper_summary.json", summary)

    if summary.failed_requests:
        logger.warning(
            "TTS verification found %d requests above WER threshold %.4f",
            summary.failed_requests,
            config.wer_threshold,
        )
    if summary.error_requests or summary.errors:
        logger.warning(
            "TTS verification completed with %d request errors and %d run errors",
            summary.error_requests,
            len(summary.errors),
        )

    if config.fail_on_threshold and summary.failed_requests:
        raise TTSVerificationError(
            f"{summary.failed_requests} TTS requests exceeded WER threshold "
            f"{config.wer_threshold}"
        )

    return summary


def run_tts_verification(
    config: TTSVerificationConfig,
    output_dir: str | Path,
) -> TTSVerificationSummary:
    """Run configured post-run TTS verification metrics."""
    verification_dir = Path(output_dir) / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)

    if not config.wer_enabled:
        return verify_tts_outputs(
            output_dir=output_dir,
            config=config,
            transcribe_audio=None,
        )

    with ManagedWhisperServer(config=config, log_dir=verification_dir) as server:
        return verify_tts_outputs(
            output_dir=output_dir,
            config=config,
            transcribe_audio=server.transcribe,
        )


class ManagedWhisperServer:
    """Small process manager for the local faster-whisper verification service."""

    def __init__(self, config: TTSVerificationConfig, log_dir: Path):
        self.config = config
        self.log_dir = log_dir
        self.process: Optional[subprocess.Popen] = None
        self._log_file = None
        self._log_path: Optional[Path] = None

    def __enter__(self) -> "ManagedWhisperServer":
        self.launch()
        try:
            self.wait_for_ready()
        except Exception:
            self.shutdown()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    def launch(self) -> None:
        if _port_in_use(self.config.host, self.config.port):
            raise TTSVerificationError(
                f"Whisper verifier port is already in use: "
                f"{self.config.host}:{self.config.port}"
            )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self._log_path = self.log_dir / f"whisper_server_{timestamp}.log"
        self._log_file = open(self._log_path, "w", encoding="utf-8")

        command = [
            _python_executable(self.config.env_path),
            "-m",
            "veeksha.verification.whisper_server",
            "--model",
            self.config.model,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--device",
            self.config.device,
            "--compute-type",
            self.config.compute_type,
        ]

        env = os.environ.copy()
        if self.config.env_path:
            bin_dir = Path(self.config.env_path) / (
                "Scripts" if os.name == "nt" else "bin"
            )
            env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
        if self.config.gpu_ids is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.config.gpu_ids))

        logger.info("Launching Whisper verifier: %s", " ".join(command))
        self.process = subprocess.Popen(
            command,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def wait_for_ready(self) -> None:
        deadline = time.time() + self.config.startup_timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                logs = self._read_recent_logs()
                raise TTSVerificationError(
                    "Whisper verifier exited before becoming ready"
                    + (f":\n{logs}" if logs else "")
                )
            try:
                response = requests.get(self._health_url, timeout=5)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(self.config.health_check_interval)

        logs = self._read_recent_logs()
        raise TTSVerificationError(
            f"Whisper verifier did not become ready within "
            f"{self.config.startup_timeout}s" + (f":\n{logs}" if logs else "")
        )

    def transcribe(self, audio_path: Path) -> str:
        with audio_path.open("rb") as audio_file:
            response = requests.post(
                self._transcribe_url,
                files={"file": (audio_path.name, audio_file, "audio/wav")},
                timeout=self.config.request_timeout,
            )
        response.raise_for_status()
        data = response.json()
        transcript = data.get("text")
        if not isinstance(transcript, str):
            raise TTSVerificationError(
                f"Whisper verifier returned invalid transcript payload: {data}"
            )
        return transcript

    def shutdown(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Whisper verifier did not shut down; killing it")
                self.process.kill()
                self.process.wait(timeout=5)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    @property
    def _health_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/health"

    @property
    def _transcribe_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/transcribe"

    def _read_recent_logs(self, lines: int = 50) -> str:
        if self._log_file is not None:
            try:
                self._log_file.flush()
            except Exception:
                pass
        if self._log_path is None or not self._log_path.exists():
            return ""
        return "\n".join(
            self._log_path.read_text(errors="replace").splitlines()[-lines:]
        )


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _save_rows(path: Path, rows: list[TTSVerificationRow]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")


def _save_summary(path: Path, summary: TTSVerificationSummary) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)


def _build_summary(
    rows: list[TTSVerificationRow],
    config: TTSVerificationConfig,
    errors: list[str],
) -> TTSVerificationSummary:
    wers = [row.wer for row in rows if row.wer is not None]
    utmos_values = [row.utmos for row in rows if row.utmos is not None]
    passed_requests = sum(1 for row in rows if row.passed is True)
    failed_requests = sum(1 for row in rows if row.passed is False)
    error_requests = sum(1 for row in rows if row.error is not None)
    utmos_failed = sum(1 for row in rows if row.utmos_error is not None)
    return TTSVerificationSummary(
        total_requests=len(rows),
        transcribed_requests=len(wers),
        passed_requests=passed_requests,
        failed_requests=failed_requests,
        error_requests=error_requests,
        wer_avg=(sum(wers) / len(wers)) if wers else None,
        wer_p50=_percentile(wers, 0.50),
        wer_p90=_percentile(wers, 0.90),
        wer_p99=_percentile(wers, 0.99),
        wer_max=max(wers) if wers else None,
        wer_threshold=config.wer_threshold,
        fail_on_threshold=config.fail_on_threshold,
        wer_enabled=config.wer_enabled,
        utmos_enabled=config.utmos_enabled,
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


def _python_executable(env_path: Optional[str]) -> str:
    if env_path is None:
        return sys.executable
    bin_dir = Path(env_path) / ("Scripts" if os.name == "nt" else "bin")
    return str(bin_dir / ("python.exe" if os.name == "nt" else "python"))


def _port_in_use(host: str, port: int) -> bool:
    try:
        addr_info = socket.getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror:
        return False

    for family, socktype, proto, _, sockaddr in addr_info:
        try:
            with socket.socket(family, socktype, proto) as sock:
                sock.settimeout(1.0)
                if sock.connect_ex(sockaddr) == 0:
                    return True
        except OSError:
            continue
    return False
