"""Post-run TTS verification with Whisper transcription and WER."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import requests

from veeksha.config.verification import TTSVerificationConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


TranscribeFn = Callable[[Path], str]


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
    error: Optional[str] = None


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
    errors: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize reference and transcript text before WER."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute WER with jiwer when available, with a small fallback for tests."""
    try:
        import jiwer

        transform = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
        return float(
            jiwer.wer(
                reference,
                hypothesis,
                reference_transform=transform,
                hypothesis_transform=transform,
            )
        )
    except ImportError:
        pass

    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return _edit_distance(reference_words, hypothesis_words) / len(reference_words)


def verify_tts_outputs(
    output_dir: str | Path,
    config: TTSVerificationConfig,
    transcribe_audio: TranscribeFn,
) -> TTSVerificationSummary:
    """Verify saved TTS audio files and persist JSON artifacts."""
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
            row_error: Optional[str] = None
            transcript = ""
            wer: Optional[float] = None
            passed: Optional[bool] = None

            if not reference_text:
                row_error = "Missing input_text in request-level metrics"
            elif not audio_path.exists():
                row_error = f"Missing audio file: {audio_path}"
            else:
                try:
                    transcript = transcribe_audio(audio_path)
                    wer = compute_wer(reference_text, transcript)
                    passed = wer <= config.wer_threshold
                except Exception as exc:
                    row_error = f"Transcription failed: {exc}"

            rows.append(
                TTSVerificationRow(
                    request_id=int(request_id),
                    reference_text=reference_text,
                    transcript=transcript,
                    wer=wer,
                    passed=passed,
                    audio_path=str(audio_path),
                    error=row_error,
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
    """Launch the managed Whisper service and verify saved TTS outputs."""
    verification_dir = Path(output_dir) / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)

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
            f"{self.config.startup_timeout}s"
            + (f":\n{logs}" if logs else "")
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
        return "\n".join(self._log_path.read_text(errors="replace").splitlines()[-lines:])


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
    passed_requests = sum(1 for row in rows if row.passed is True)
    failed_requests = sum(1 for row in rows if row.passed is False)
    error_requests = sum(1 for row in rows if row.error is not None)
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
        errors=errors,
    )


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
