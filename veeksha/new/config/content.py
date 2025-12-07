from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.new.config.channels import BaseChannelConfig
from veeksha.types.base_int_enum import BaseIntEnum
from veeksha.types.base_registry import BaseRegistry


# ----- Types -----
class ContentType(BaseIntEnum):
    SYNTHETIC = 1
    TRACE = 2
    LMEVAL = 3


# ----- Configs -----
@frozen_dataclass
class ContentConfig(BasePolyConfig):
    pass


@frozen_dataclass
class SyntheticContentConfig(ContentConfig):
    channels: list[BaseChannelConfig] = field(
        default_factory=list,
        metadata={"help": "The modality channels for the synthetic content."},
    )

    @classmethod
    def get_type(cls):
        return ContentType.SYNTHETIC

    def __post_init__(self):
        channel_types = set([channel.get_type() for channel in self.channels])
        if len(channel_types) != len(self.channels):
            raise ValueError("All channels must have unique types")


@frozen_dataclass
class TraceContentConfig(ContentConfig):
    pass

    @classmethod
    def get_type(cls):
        return ContentType.TRACE


@frozen_dataclass
class LmevalContentConfig(ContentConfig):
    pass

    @classmethod
    def get_type(cls):
        return ContentType.LMEVAL


# ----- Registry -----
class ContentRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ContentType:
        return ContentType.from_str(key_str)  # type: ignore


ContentRegistry.register(ContentType.SYNTHETIC, SyntheticContentConfig)
ContentRegistry.register(ContentType.TRACE, TraceContentConfig)
ContentRegistry.register(ContentType.LMEVAL, LmevalContentConfig)
