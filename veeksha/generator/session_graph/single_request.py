from veeksha.config.generator.session_graph import (
    SingleRequestSessionGraphGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.core.session_graph import SessionGraph, SessionNode, add_node
from veeksha.generator.session_graph.base import BaseSessionGraphGenerator


class SingleRequestSessionGraphGenerator(BaseSessionGraphGenerator):
    def __init__(
        self,
        config: SingleRequestSessionGraphGeneratorConfig,
        seed_manager: SeedManager,
    ):
        self.config = config
        self.seed_manager = seed_manager

    def generate_session_graph(self) -> SessionGraph:
        session_graph = SessionGraph()
        add_node(session_graph, SessionNode(id=0, wait_after_ready=0))
        return session_graph
