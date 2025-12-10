from abc import abstractmethod

from veeksha.new.config.generator.session_graph import BaseSessionGraphGeneratorConfig

# from veeksha.new.core.session_graph import SessionGraph # TODO: implement


class BaseSessionGraphGenerator:
    def __init__(self, config: BaseSessionGraphGeneratorConfig):
        self.config = config

    @abstractmethod
    def generate_session_graph(self):
        pass
