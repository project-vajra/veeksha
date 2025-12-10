from veeksha.new.config.generator.session_graph import LinearSessionGraphGeneratorConfig
from veeksha.new.generator.session_graph.base import BaseSessionGraphGenerator

# from veeksha.new.core.session_graph import SessionGraph # TODO: implement


class LinearSessionGraphGenerator(BaseSessionGraphGenerator):
    def __init__(self, config: LinearSessionGraphGeneratorConfig):
        self.config = config
        self.request_wait_generator = config.request_wait_generator
        self.num_request_generator = config.num_request_generator

    # TODO: implement
    def generate_session_graph(self):
        pass
