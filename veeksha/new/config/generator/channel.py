from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.length_generator import (
    UniformRequestLengthGeneratorConfig,
)
from veeksha.config.generators.length_generator.base_generator import (
    BaseRequestLengthGeneratorConfig,
)
from veeksha.new.types import ChannelModality
from veeksha.types.base_registry import BaseRegistry


@frozen_dataclass
class BaseChannelGeneratorConfig(BasePolyConfig):
    pass


@frozen_dataclass
class TextChannelGeneratorConfig(BaseChannelGeneratorConfig):

    length_generator: BaseRequestLengthGeneratorConfig = field(
        default_factory=UniformRequestLengthGeneratorConfig
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
