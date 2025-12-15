import numpy as np

from veeksha.new.config.generator.channel import TextChannelGeneratorConfig
from veeksha.new.generator.channel.base import BaseChannelGenerator


class TextChannelGenerator(BaseChannelGenerator):
    def __init__(self, config: TextChannelGeneratorConfig, rng: np.random.RandomState):
        self.config = config
        self.rng = rng

    def generate_content(self):
        pass
