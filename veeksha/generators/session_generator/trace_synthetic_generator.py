from veeksha.generators.session_generator.base_generator import BaseSessionGenerator
from veeksha.config.generators.session_generator.trace_synthetic_generator import TraceSyntheticSessionGeneratorConfig
from veeksha.types.session_generator_type import SessionGeneratorType

class TraceSyntheticSessionGenerator(BaseSessionGenerator):
    def __init__(
        self,
        config: TraceSyntheticSessionGeneratorConfig,
    ):
        self.config = config