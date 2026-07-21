"""Evaluator configuration classes.

This module defines the configuration hierarchy for evaluators in the new veeksha
framework. Evaluators are responsible for computing metrics from benchmark runs.

The hierarchy follows the BasePolyConfig pattern used elsewhere in veeksha:
- BaseEvaluatorConfig (abstract base)
  - PerformanceEvaluatorConfig (latency, throughput)
  - LMEvalAccuracyEvaluatorConfig (task-specific correctness - lm-eval)
  - AudioQualityEvaluatorConfig (generated audio quality checks)
"""

from typing import List, Optional, Union, cast

from vidhi import BasePolyConfig, field, frozen_dataclass

from veeksha.config.slo import BaseSloConfig, ConstantSloConfig
from veeksha.config.verification import AudioVerificationConfig
from veeksha.types import ChannelModality, EvaluationType


@frozen_dataclass
class DecodeWindowConfig:
    """Configuration for decode window analysis.

    Filters metrics to the window where the server is doing decode
    with a full batch of requests.
    """

    min_active_requests: Union[int, str] = field(
        1,
        help="Minimum number of simultaneously generating (decoding) requests "
        "required for a time interval to be considered inside the decode window. "
        "Use 'max_observed' to auto-detect the peak concurrent decoding count.",
    )
    selection_strategy: str = field(
        "longest",
        help="Which window(s) to analyze when multiple windows exist. "
        "Supported: 'longest' (single longest), 'first' (single first), "
        "'all' (aggregate all qualifying windows).",
    )
    anchor_to_client_pickup: bool = field(
        True,
        help="If True, anchor per-request token times to client_picked_up_at "
        "when available; otherwise use scheduler_dispatched_at.",
    )
    require_streaming: bool = field(
        True,
        help="If True, only streaming requests contribute to decode window analysis.",
    )

    def __post_init__(self):
        if isinstance(self.min_active_requests, int):
            if self.min_active_requests <= 0:
                raise ValueError("min_active_requests must be > 0")
        elif isinstance(self.min_active_requests, str):
            if self.min_active_requests != "max_observed":
                raise ValueError(
                    f"Invalid min_active_requests '{self.min_active_requests}'. "
                    "Supported string value: 'max_observed'"
                )
        else:
            raise ValueError("min_active_requests must be int or 'max_observed'")
        allowed = {"longest", "first", "all"}
        if self.selection_strategy not in allowed:
            raise ValueError(
                f"Invalid selection_strategy '{self.selection_strategy}'. "
                f"Supported: {sorted(allowed)}"
            )


@frozen_dataclass
class BaseChannelPerformanceConfig(BasePolyConfig):
    """Base config for channel-specific performance"""


@frozen_dataclass
class TextChannelPerformanceConfig(BaseChannelPerformanceConfig):
    """Text channel performance configuration"""

    decode_window_enabled: bool = field(False, help="Enable decode window analysis")
    decode_window_config: Optional[DecodeWindowConfig] = field(
        None, help="Decode window configuration (required if enabled)"
    )

    @classmethod
    def get_type(cls) -> ChannelModality:
        return ChannelModality.TEXT

    def __post_init__(self):
        if self.decode_window_enabled and self.decode_window_config is None:
            raise ValueError(
                "decode_window_config is required when decode_window_enabled=True"
            )


class ImageChannelPerformanceConfig(BaseChannelPerformanceConfig):
    """Image channel performance configuration"""

    @classmethod
    def get_type(cls) -> ChannelModality:
        return ChannelModality.IMAGE


@frozen_dataclass
class AudioChannelPerformanceConfig(BaseChannelPerformanceConfig):
    """Audio channel performance configuration for TTS benchmarking."""

    interactivity_enabled: bool = field(
        True,
        help="Compute streaming interactivity metrics when timestamp lists are present.",
    )
    startup_delay_ms_values: List[float] = field(
        default_factory=lambda: [0.0, 100.0, 300.0],
        help="Fixed delays after first audio to simulate, in milliseconds.",
    )
    startup_buffer_ms_values: List[float] = field(
        default_factory=lambda: [0.0, 100.0, 300.0],
        help="Playable-audio targets to simulate before playback, in milliseconds.",
    )
    min_reportable_stall_ms: float = field(
        10.0,
        help="Playback gaps at or below this duration are treated as transport noise.",
    )
    fluidity_frame_ms: float = field(
        20.0,
        help=(
            "Duration of each complete playable PCM frame used by the "
            "Etalon-inspired audio fluidity metric."
        ),
    )
    fluidity_startup_delay_ms: float = field(
        0.0,
        help=(
            "Playback slack after the first complete frame for the primary "
            "user_audio_fluidity_index metric. Set this explicitly when the "
            "playback client intentionally buffers before starting."
        ),
    )
    fluidity_attribution_mode: str = field(
        "conservative",
        help=(
            "How to attribute fluidity misses to the TTS service. "
            "'conservative' emits tts_service_fluidity_index only when all "
            "text was sent before the first playable frame. "
            "'source_oversupplied' treats every miss as service-caused and must "
            "only be used with a trace that guarantees synthesis-eligible text "
            "throughout playback. User fluidity is always reported."
        ),
    )
    persist_raw_timing: bool = field(
        False,
        help="Write metrics/audio_raw_timing.jsonl with raw per-event timestamps.",
    )

    @classmethod
    def get_type(cls) -> ChannelModality:
        return ChannelModality.AUDIO

    def __post_init__(self):
        for field_name in ("startup_delay_ms_values", "startup_buffer_ms_values"):
            values = getattr(self, field_name)
            if any(value < 0 for value in values):
                raise ValueError(f"{field_name} must contain only values >= 0")
            normalized = list(dict.fromkeys(float(value) for value in values))
            if 0.0 not in normalized:
                normalized.insert(0, 0.0)
            object.__setattr__(self, field_name, normalized)
        if self.min_reportable_stall_ms < 0:
            raise ValueError("min_reportable_stall_ms must be >= 0")
        if self.fluidity_frame_ms <= 0:
            raise ValueError("fluidity_frame_ms must be > 0")
        if self.fluidity_startup_delay_ms < 0:
            raise ValueError("fluidity_startup_delay_ms must be >= 0")
        if self.fluidity_attribution_mode not in (
            "conservative",
            "source_oversupplied",
        ):
            raise ValueError(
                "fluidity_attribution_mode must be one of "
                "('conservative', 'source_oversupplied')"
            )


@frozen_dataclass
class VideoChannelPerformanceConfig(BaseChannelPerformanceConfig):
    """Video channel performance configuration"""

    @classmethod
    def get_type(cls) -> ChannelModality:
        return ChannelModality.VIDEO


def _default_slos() -> list[BaseSloConfig]:
    return [
        ConstantSloConfig(
            metric="ttfc",
            percentile=0.99,
            value=0.5,
            name="P99 TTFC",
        ),
        ConstantSloConfig(
            metric="tbc",
            percentile=0.9,
            value=0.05,
            name="P90 TBC",
        ),
    ]


def _normalize_channel_modality(channel: object) -> ChannelModality:
    if isinstance(channel, ChannelModality):
        return channel
    if isinstance(channel, str):
        return cast(ChannelModality, ChannelModality.from_str(channel))
    if isinstance(channel, int) and not isinstance(channel, bool):
        return ChannelModality(channel)
    raise ValueError(f"Invalid target channel modality: {channel!r}")


@frozen_dataclass
class BaseEvaluatorConfig(BasePolyConfig):
    """Base configuration for all evaluators (performance, accuracy)"""

    target_channels: list = field(
        default_factory=lambda: ["text"],
        help="List of ChannelModality values to evaluate.",
    )
    slos: list[BaseSloConfig] = field(
        default_factory=_default_slos,
        help="List of SLO definitions to evaluate against request-level metrics.",
    )
    stream_metrics: bool = field(True, help="Enable real-time metric streaming")
    stream_metrics_interval: float = field(
        5.0, help="Interval for streaming metrics in seconds"
    )

    def __post_init__(self):
        if self.target_channels:
            object.__setattr__(
                self,
                "target_channels",
                [_normalize_channel_modality(ch) for ch in self.target_channels],
            )


@frozen_dataclass
class PerformanceEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for system performance evaluation."""

    text_channel: TextChannelPerformanceConfig = field(
        default_factory=TextChannelPerformanceConfig,
        help="Text channel performance configuration",
    )
    image_channel: ImageChannelPerformanceConfig = field(
        default_factory=ImageChannelPerformanceConfig,
        help="Image channel performance configuration",
    )
    audio_channel: Optional[AudioChannelPerformanceConfig] = field(
        None, help="Audio channel performance configuration"
    )
    video_channel: Optional[VideoChannelPerformanceConfig] = field(
        None, help="Video channel performance configuration"
    )

    @classmethod
    def get_type(cls) -> EvaluationType:
        return EvaluationType.PERFORMANCE

    def __post_init__(self):
        super().__post_init__()
        if ChannelModality.AUDIO in self.target_channels and self.audio_channel is None:
            object.__setattr__(self, "audio_channel", AudioChannelPerformanceConfig())
        if ChannelModality.VIDEO in self.target_channels and self.video_channel is None:
            object.__setattr__(self, "video_channel", VideoChannelPerformanceConfig())

    def get_channel_config(
        self, channel: ChannelModality | str | int
    ) -> Optional[BaseChannelPerformanceConfig]:
        """Get the performance config for a specific channel."""
        channel = _normalize_channel_modality(channel)
        if channel == ChannelModality.TEXT:
            return self.text_channel
        elif channel == ChannelModality.IMAGE:
            return self.image_channel
        elif channel == ChannelModality.AUDIO:
            return self.audio_channel
        elif channel == ChannelModality.VIDEO:
            return self.video_channel
        return None


@frozen_dataclass
class LMEvalAccuracyEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for lm-eval accuracy evaluation (task-specific correctness)."""

    slos: list[BaseSloConfig] = field(
        default_factory=list,
        help="Accuracy evaluators do not use performance SLOs.",
    )
    stream_metrics: bool = field(
        False,
        help="Accuracy evaluation does not stream incremental metrics.",
    )
    bootstrap_iters: int = field(
        100000, help="Bootstrap iterations for confidence intervals"
    )

    @classmethod
    def get_type(cls) -> EvaluationType:
        return EvaluationType.ACCURACY_LMEVAL


@frozen_dataclass
class AudioQualityEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for generated audio quality evaluation."""

    target_channels: list = field(
        default_factory=lambda: ["audio"],
        help="List of modalities whose output quality should be evaluated.",
    )
    slos: list[BaseSloConfig] = field(
        default_factory=list,
        help="Accuracy evaluators do not use performance SLOs.",
    )
    stream_metrics: bool = field(
        False,
        help="Accuracy evaluation does not stream incremental metrics.",
    )
    verification: AudioVerificationConfig = field(
        default_factory=AudioVerificationConfig,
        help="Generated audio verification configuration.",
    )
    save_audio_files: bool = field(
        True,
        help="Whether to persist audio artifacts for quality evaluation.",
    )

    @classmethod
    def get_type(cls) -> EvaluationType:
        return EvaluationType.AUDIO_QUALITY
