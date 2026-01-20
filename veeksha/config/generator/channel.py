from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generator.length import (
    BaseLengthGeneratorConfig,
    UniformLengthGeneratorConfig,
)
from veeksha.types import ChannelModality, LengthGeneratorType
from veeksha.types.base_registry import BaseRegistry


@frozen_dataclass
class BaseChannelGeneratorConfig(BasePolyConfig):
    pass


@frozen_dataclass
class TextChannelGeneratorConfig(BaseChannelGeneratorConfig):
    """Configuration for text channel input generation."""

    body_length_generator: BaseLengthGeneratorConfig = field(
        default_factory=UniformLengthGeneratorConfig,
        metadata={
            "help": f"The generator for the body (prompt) length. {LengthGeneratorType.help_str()}"
        },
    )
    shared_prefix_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Fraction of prompt tokens to use as shared prefix for root requests (0.0-1.0)"
        },
    )
    shared_prefix_probability: float = field(
        default=1.0,
        metadata={
            "help": "Probability that a root request uses shared prefix (0.0-1.0)"
        },
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
    def __post_init__(self):
        raise NotImplementedError("AudioChannelConfig is not implemented")

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
