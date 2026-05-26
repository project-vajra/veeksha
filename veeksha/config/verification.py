from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class TTSVerificationConfig:
    """Post-run TTS verification for generated audio quality."""

    enabled: bool = field(
        default=False,
        metadata={"help": "Enable post-run TTS transcription/WER verification."},
    )
    wer_enabled: bool = field(
        default=True,
        metadata={"help": "Enable WER verification using the managed Whisper service."},
    )
    utmos_enabled: bool = field(
        default=False,
        metadata={"help": "Enable UTMOS predicted MOS scoring for generated audio."},
    )
    model: str = field(
        default="large-v3",
        metadata={"help": "Whisper model identifier passed to faster-whisper."},
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

    utmos_hf_repo: str = field(
        default="balacoon/utmos",
        metadata={"help": "Hugging Face model repo containing the UTMOS JIT file."},
    )
    utmos_jit_file: str = field(
        default="utmos.jit",
        metadata={"help": "TorchScript filename to load from the UTMOS HF repo."},
    )
    utmos_device: str = field(
        default="cpu",
        metadata={
            "help": "Device for UTMOS TorchScript inference, e.g. 'cpu' or 'cuda:0'."
        },
    )

    def __post_init__(self):
        if self.enabled and not (self.wer_enabled or self.utmos_enabled):
            raise ValueError(
                "TTSVerificationConfig requires wer_enabled or utmos_enabled when enabled=True."
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
        if not self.utmos_hf_repo:
            raise ValueError("TTSVerificationConfig.utmos_hf_repo is required")
        if not self.utmos_jit_file:
            raise ValueError("TTSVerificationConfig.utmos_jit_file is required")
        if not self.utmos_device:
            raise ValueError("TTSVerificationConfig.utmos_device is required")
