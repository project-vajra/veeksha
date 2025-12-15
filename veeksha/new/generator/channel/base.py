from abc import abstractmethod

from veeksha.new.config.generator.channel import BaseChannelGeneratorConfig


class BaseChannelGenerator:
    def __init__(self, config: BaseChannelGeneratorConfig, rng: None):
        self.config = config
        self.rng = rng

    @abstractmethod
    def generate_content(self):
        pass
