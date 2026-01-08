from veeksha.generator.channel.text import TextChannelGenerator
from veeksha.types import ChannelModality
from veeksha.types.base_registry import BaseRegistry


class ChannelGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ChannelModality:
        return ChannelModality.from_str(key_str)  # type: ignore


ChannelGeneratorRegistry.register(ChannelModality.TEXT, TextChannelGenerator)
