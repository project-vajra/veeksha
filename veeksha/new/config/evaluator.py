"""Evaluator configuration classes.

This module defines the configuration hierarchy for evaluators in the new veeksha
framework. Evaluators are responsible for computing metrics from benchmark runs.

The hierarchy follows the BasePolyConfig pattern used elsewhere in veeksha:
- BaseEvaluatorConfig (abstract base)
  - PerformanceEvaluatorConfig (latency, throughput)
  - LMEvalAccuracyEvaluatorConfig (task-specific correctness - lm-eval)
"""

from dataclasses import field
from typing import Optional

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.new.config.core.frozen_dataclass import frozen_dataclass
from veeksha.new.config.slo import BaseSloConfig, ConstantSloConfig
from veeksha.new.types import ChannelModality, EvaluationType


@frozen_dataclass
class DecodeWindowConfig:
    """Configuration for decode window analysis.

    Filters metrics to the window where the server is doing decode
    with a full batch of requests.
    """

    # TODO


# ---- Channel-specific performance configs ----


@frozen_dataclass
class BaseChannelPerformanceConfig(BasePolyConfig):
    """Base config for channel-specific performance"""


@frozen_dataclass
class TextChannelPerformanceConfig(BaseChannelPerformanceConfig):
    """Text channel performance configuration"""

    decode_window_enabled: bool = field(
        default=False, metadata={"help": "Enable decode window analysis"}
    )
    decode_window_config: Optional[DecodeWindowConfig] = field(
        default=None,
        metadata={"help": "Decode window configuration (required if enabled)"},
    )

    @classmethod
    def get_type(cls) -> ChannelModality:
        return ChannelModality.TEXT


class ImageChannelPerformanceConfig(BaseChannelPerformanceConfig):
    """Image channel performance configuration"""

    @classmethod
    def get_type(cls) -> ChannelModality:
        return ChannelModality.IMAGE


@frozen_dataclass
class AudioChannelPerformanceConfig(BaseChannelPerformanceConfig):
    """Audio channel performance configuration"""

    @classmethod
    def get_type(cls) -> ChannelModality:
        return ChannelModality.AUDIO


@frozen_dataclass
class VideoChannelPerformanceConfig(BaseChannelPerformanceConfig):
    """Video channel performance configuration"""

    @classmethod
    def get_type(cls) -> ChannelModality:
        return ChannelModality.VIDEO


# ---- Base evaluator config ----


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


@frozen_dataclass
class BaseEvaluatorConfig(BasePolyConfig):
    """Base configuration for all evaluators (performance, accuracy)"""

    target_channels: list = field(
        default_factory=lambda: ["text"],
        metadata={"help": "List of ChannelModality values to evaluate."},
    )

    slos: list[BaseSloConfig] = field(
        default_factory=_default_slos,
        metadata={
            "help": "List of SLO definitions to evaluate against request-level metrics."
        },
    )

    stream_metrics: bool = field(
        default=True, metadata={"help": "Enable real-time metric streaming"}
    )
    stream_metrics_interval: float = field(
        default=5.0, metadata={"help": "Interval for streaming metrics in seconds"}
    )

    wandb_enabled: bool = field(
        default=False, metadata={"help": "Enable Weights & Biases logging"}
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "WandB project name"}
    )
    wandb_group: Optional[str] = field(
        default=None, metadata={"help": "WandB group name"}
    )
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "WandB run name"}
    )

    def __post_init__(self):
        if self.target_channels:
            converted = []
            for ch in self.target_channels:
                if isinstance(ch, str):
                    converted.append(ChannelModality.from_str(ch))
                else:
                    converted.append(ch)
            object.__setattr__(self, "target_channels", converted)


# ---- Performance evaluator config ----


@frozen_dataclass
class PerformanceEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for system performance evaluation."""

    text_channel: TextChannelPerformanceConfig = field(
        default_factory=TextChannelPerformanceConfig,
        metadata={"help": "Text channel performance configuration"},
    )
    image_channel: ImageChannelPerformanceConfig = field(
        default_factory=ImageChannelPerformanceConfig,
        metadata={"help": "Image channel performance configuration"},
    )
    audio_channel: Optional[AudioChannelPerformanceConfig] = field(
        default=None,
        metadata={"help": "Audio channel performance configuration"},
    )
    video_channel: Optional[VideoChannelPerformanceConfig] = field(
        default=None,
        metadata={"help": "Video channel performance configuration"},
    )

    @classmethod
    def get_type(cls) -> EvaluationType:
        return EvaluationType.PERFORMANCE

    def get_channel_config(
        self, channel: ChannelModality
    ) -> Optional[BaseChannelPerformanceConfig]:
        """Get the performance config for a specific channel."""
        if channel == ChannelModality.TEXT:
            return self.text_channel
        elif channel == ChannelModality.IMAGE:
            return self.image_channel
        elif channel == ChannelModality.AUDIO:
            return self.audio_channel
        elif channel == ChannelModality.VIDEO:
            return self.video_channel
        return None


# ---- Accuracy evaluator config(s) ----


@frozen_dataclass
class LMEvalAccuracyEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for lm-eval accuracy evaluation (task-specific correctness).

    IMPORTANT: For lm-eval accuracy evaluation, the content generation must use
    `LMEvalSessionGenerator`. The generator owns the lm-eval Task/Instance objects,
    and the evaluator binds responses to instances for evaluation.
    """

    bootstrap_iters: int = field(
        default=100000,
        metadata={"help": "Bootstrap iterations for confidence intervals"},
    )

    @classmethod
    def get_type(cls) -> EvaluationType:
        return EvaluationType.ACCURACY
