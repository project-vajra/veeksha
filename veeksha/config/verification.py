from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class TTSVerificationConfig:
    """Post-run TTS verification using a managed local Whisper ASR service."""

    enabled: bool = field(
        default=False,
        metadata={"help": "Enable post-run TTS transcription/WER verification."},
    )
    backend: str = field(
        default="faster_whisper",
        metadata={
            "help": "ASR backend for verification. Supported: 'faster_whisper'."
        },
    )
    model: str = field(
        default="large-v3",
        metadata={
            "help": "Whisper model identifier passed to faster-whisper."
        },
    )
    device: str = field(
        default="cuda",
        metadata={"help": "Device passed to faster-whisper, e.g. 'cuda' or 'cpu'."},
    )
    compute_type: str = field(
        default="float16",
        metadata={
            "help": "Compute type passed to faster-whisper, e.g. 'float16' or 'int8'."
        },
    )
    host: str = field(
        default="localhost",
        metadata={"help": "Host for the managed Whisper verification service."},
    )
    port: int = field(
        default=8077,
        metadata={"help": "Port for the managed Whisper verification service."},
    )
    startup_timeout: int = field(
        default=300,
        metadata={"help": "Seconds to wait for the Whisper service to become ready."},
    )
    health_check_interval: float = field(
        default=2.0,
        metadata={"help": "Seconds between Whisper service health probes."},
    )
    request_timeout: int = field(
        default=300,
        metadata={"help": "Seconds to wait for each transcription request."},
    )
    wer_threshold: float = field(
        default=0.05,
        metadata={"help": "Per-request WER threshold for pass/fail classification."},
    )
    fail_on_threshold: bool = field(
        default=False,
        metadata={
            "help": "If True, raise after saving artifacts when any request exceeds the WER threshold."
        },
    )
    env_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional virtualenv/conda environment path for launching the verifier service."
        },
    )
    gpu_ids: Optional[list[int]] = field(
        default=None,
        metadata={
            "help": "Optional GPU IDs for the verifier service CUDA_VISIBLE_DEVICES."
        },
    )

    def __post_init__(self):
        if self.backend != "faster_whisper":
            raise ValueError(
                f"Unsupported TTS verification backend: {self.backend}. "
                "Supported: faster_whisper"
            )
        if self.port <= 0:
            raise ValueError("TTSVerificationConfig.port must be > 0")
        if self.startup_timeout <= 0:
            raise ValueError("TTSVerificationConfig.startup_timeout must be > 0")
        if self.health_check_interval <= 0:
            raise ValueError("TTSVerificationConfig.health_check_interval must be > 0")
        if self.request_timeout <= 0:
            raise ValueError("TTSVerificationConfig.request_timeout must be > 0")
        if self.wer_threshold < 0:
            raise ValueError("TTSVerificationConfig.wer_threshold must be >= 0")
