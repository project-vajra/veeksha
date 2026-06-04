from vidhi import BasePolyConfig, field, frozen_dataclass

from veeksha.types import VerificationType


@frozen_dataclass
class BaseVerificationConfig(BasePolyConfig):
    """Base class for post-run verification configuration."""

    fail_on_threshold: bool = field(
        False,
        help="If True, fail the run when verification thresholds fail.",
    )

    def is_enabled(self) -> bool:
        return False


@frozen_dataclass
class WhisperTranscriptionConfig:
    """Whisper transcription configuration used by WER verification."""

    model: str = field(
        "large-v3",
        help="Whisper model identifier passed to faster-whisper.",
    )
    device: str = field(
        "cuda",
        help="Device passed to faster-whisper, e.g. 'cuda' or 'cpu'.",
    )
    compute_type: str = field(
        "float16",
        help="Compute type passed to faster-whisper, e.g. 'float16' or 'int8'.",
    )

    def __post_init__(self):
        if not self.model:
            raise ValueError("WhisperTranscriptionConfig.model is required")
        if not self.device:
            raise ValueError("WhisperTranscriptionConfig.device is required")
        if not self.compute_type:
            raise ValueError("WhisperTranscriptionConfig.compute_type is required")


@frozen_dataclass
class WERVerifierConfig:
    """WER verifier configuration for generated speech."""

    enabled: bool = field(
        False,
        help="Enable WER verification using inline Whisper transcription.",
    )
    threshold: float = field(
        0.05,
        help="Per-request WER threshold for pass/fail classification.",
    )
    whisper: WhisperTranscriptionConfig = field(
        default_factory=WhisperTranscriptionConfig,
        help="Inline Whisper transcription configuration.",
    )

    def __post_init__(self):
        if self.threshold < 0:
            raise ValueError("WERVerifierConfig.threshold must be >= 0")


@frozen_dataclass
class UTMOSVerifierConfig:
    """UTMOS predicted MOS verifier configuration."""

    enabled: bool = field(
        False,
        help="Enable UTMOS predicted MOS scoring for generated audio.",
    )
    hf_repo: str = field(
        "balacoon/utmos",
        help="Hugging Face model repo containing the UTMOS JIT file.",
    )
    jit_file: str = field(
        "utmos.jit",
        help="TorchScript filename to load from the UTMOS HF repo.",
    )
    device: str = field(
        "cuda:0",
        help="Device for UTMOS TorchScript inference, e.g. 'cuda:0' or 'cpu'.",
    )

    def __post_init__(self):
        if not self.hf_repo:
            raise ValueError("UTMOSVerifierConfig.hf_repo is required")
        if not self.jit_file:
            raise ValueError("UTMOSVerifierConfig.jit_file is required")
        if not self.device:
            raise ValueError("UTMOSVerifierConfig.device is required")


@frozen_dataclass
class AudioVerificationConfig(BaseVerificationConfig):
    """Post-run verification for generated audio artifacts."""

    max_requests: int = field(
        2000,
        help="Maximum number of request rows to verify. Use 0 or less to verify all rows.",
    )
    wer: WERVerifierConfig = field(
        default_factory=WERVerifierConfig,
        help="WER verifier configuration.",
    )
    utmos: UTMOSVerifierConfig = field(
        default_factory=UTMOSVerifierConfig,
        help="UTMOS verifier configuration.",
    )

    @classmethod
    def get_type(cls) -> VerificationType:
        return VerificationType.AUDIO

    def is_enabled(self) -> bool:
        return bool(self.wer.enabled or self.utmos.enabled)
