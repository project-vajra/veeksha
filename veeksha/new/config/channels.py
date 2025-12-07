from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.length_generator import (
    UniformRequestLengthGeneratorConfig,
)
from veeksha.config.generators.length_generator.base_generator import (
    BaseRequestLengthGeneratorConfig,
)
from veeksha.types.base_int_enum import BaseIntEnum
from veeksha.types.base_registry import BaseRegistry


class ContentModality(BaseIntEnum):
    TEXT = 1
    IMAGE = 2
    AUDIO = 3
    VIDEO = 4


@frozen_dataclass
class BaseChannelConfig(BasePolyConfig):
    pass


@frozen_dataclass
class TextChannelConfig(BaseChannelConfig):

    length_generator: BaseRequestLengthGeneratorConfig = field(
        default_factory=UniformRequestLengthGeneratorConfig
    )

    @classmethod
    def get_type(cls):
        return ContentModality.TEXT


@frozen_dataclass
class ImageChannelConfig(BaseChannelConfig):

    def __post_init__(self):
        raise NotImplementedError("ImageChannelConfig is not implemented")

    @classmethod
    def get_type(cls):
        return ContentModality.IMAGE


@frozen_dataclass
class AudioChannelConfig(BaseChannelConfig):
    def __post_init__(self):
        raise NotImplementedError("AudioChannelConfig is not implemented")

    @classmethod
    def get_type(cls):
        return ContentModality.AUDIO


@frozen_dataclass
class VideoChannelConfig(BaseChannelConfig):

    def __post_init__(self):
        raise NotImplementedError("VideoChannelConfig is not implemented")

    @classmethod
    def get_type(cls):
        return ContentModality.VIDEO


# channel registry
class ChannelRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ContentModality:
        return ContentModality.from_str(key_str)  # type: ignore


ChannelRegistry.register(ContentModality.TEXT, TextChannelConfig)
ChannelRegistry.register(ContentModality.IMAGE, ImageChannelConfig)
ChannelRegistry.register(ContentModality.AUDIO, AudioChannelConfig)
ChannelRegistry.register(ContentModality.VIDEO, VideoChannelConfig)
