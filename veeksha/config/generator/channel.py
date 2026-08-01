from vidhi import BasePolyConfig, field, frozen_dataclass

from veeksha.config.generator.length import (
    BaseLengthGeneratorConfig,
    UniformLengthGeneratorConfig,
)
from veeksha.types import ChannelModality
from veeksha.types.base_registry import BaseRegistry


@frozen_dataclass
class BaseChannelGeneratorConfig(BasePolyConfig):
    """Input channel modality (text, image, audio, or video)."""


@frozen_dataclass
class TextChannelGeneratorConfig(BaseChannelGeneratorConfig):
    """Configuration for text channel input generation."""

    body_length_generator: BaseLengthGeneratorConfig = field(
        default_factory=UniformLengthGeneratorConfig,
        help="The generator for the body (prompt) length.",
    )
    shared_prefix_ratio: float = field(
        0.0,
        help="Fraction of prompt tokens to use as shared prefix for root requests (0.0-1.0)",
    )
    shared_prefix_probability: float = field(
        1.0,
        help="Probability that a root request uses shared prefix (0.0-1.0)",
    )

    @classmethod
    def get_type(cls):
        return ChannelModality.TEXT


@frozen_dataclass
class ImageChannelGeneratorConfig(BaseChannelGeneratorConfig):

    def __post_init__(self):
        raise NotImplementedError("ImageChannelConfig is not implemented")

    @classmethod
    def get_type(cls):
        return ChannelModality.IMAGE


@frozen_dataclass
class AudioChannelGeneratorConfig(BaseChannelGeneratorConfig):
    """Synthetic audio input: a deterministic generated waveform clip.

    Writes a WAV (cached per parameter set) and hands its path to the client as
    audio input. Content is synthetic (a tone or silence) -- useful for STT
    smoke-testing and preflight without a real audio dataset.
    """

    duration_seconds: float = field(
        3.0, help="Length of the generated audio clip in seconds."
    )
    sample_rate: int = field(16000, help="Audio sample rate in Hz.")
    waveform: str = field("sine", help="Waveform to synthesize: 'sine' or 'silence'.")
    frequency_hz: float = field(
        440.0, help="Tone frequency in Hz (used when waveform='sine')."
    )

    def __post_init__(self):
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.waveform not in ("sine", "silence"):
            raise ValueError("waveform must be 'sine' or 'silence'")

    @classmethod
    def get_type(cls):
        return ChannelModality.AUDIO


@frozen_dataclass
class VideoChannelGeneratorConfig(BaseChannelGeneratorConfig):

    def __post_init__(self):
        raise NotImplementedError("VideoChannelConfig is not implemented")

    @classmethod
    def get_type(cls):
        return ChannelModality.VIDEO


# channel registry
class ChannelGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ChannelModality:
        return ChannelModality.from_str(key_str)  # type: ignore


ChannelGeneratorRegistry.register(ChannelModality.TEXT, TextChannelGeneratorConfig)
ChannelGeneratorRegistry.register(ChannelModality.IMAGE, ImageChannelGeneratorConfig)
ChannelGeneratorRegistry.register(ChannelModality.AUDIO, AudioChannelGeneratorConfig)
ChannelGeneratorRegistry.register(ChannelModality.VIDEO, VideoChannelGeneratorConfig)
