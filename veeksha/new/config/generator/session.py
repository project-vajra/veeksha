from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.new.config.generator.channel import BaseChannelGeneratorConfig
from veeksha.new.config.generator.session_graph import (
    BaseSessionGraphGeneratorConfig,
    LinearSessionGraphGeneratorConfig,
)
from veeksha.new.types import SessionGeneratorType


@frozen_dataclass
class BaseSessionGeneratorConfig(BasePolyConfig):
    # TODO: how to think about system prompts?
    pass


@frozen_dataclass
class SyntheticSessionGeneratorConfig(BaseSessionGeneratorConfig):
    session_graph: BaseSessionGraphGeneratorConfig = field(
        default_factory=LinearSessionGraphGeneratorConfig,
        metadata={"help": "The generator for the session graphs. Available: linear"},
    )
    channels: list[BaseChannelGeneratorConfig] = field(
        default_factory=list,
        metadata={
            "help": "The modality channels for the content of each request. Available: text"
        },
    )

    @classmethod
    def get_type(cls):
        return SessionGeneratorType.SYNTHETIC

    def __post_init__(self):
        channel_types = set([channel.get_type() for channel in self.channels])
        if len(channel_types) != len(self.channels):
            raise ValueError("All channel generators must have unique types")


@frozen_dataclass
class LmevalSessionGeneratorConfig(BaseSessionGeneratorConfig):
    pass

    @classmethod
    def get_type(cls):
        return SessionGeneratorType.LMEVAL
