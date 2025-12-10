from veeksha.new.config.generator.channel import TextChannelGeneratorConfig
from veeksha.new.generator.channel.base import BaseChannelGenerator


class TextChannelGenerator(BaseChannelGenerator):
    def __init__(self, config: TextChannelGeneratorConfig):
        self.config = config

    def generate_content(self):
        pass
