from abc import abstractmethod

from veeksha.new.config.generator.channel import BaseChannelGeneratorConfig


class BaseChannelGenerator:
    def __init__(self, config: BaseChannelGeneratorConfig):
        self.config = config

    @abstractmethod
    def generate_content(self):
        pass
