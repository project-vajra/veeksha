from veeksha.config.generator.session_graph import LinearSessionGraphGeneratorConfig
from veeksha.core.seeding import SeedManager
from veeksha.core.session_graph import (
    SessionEdge,
    SessionGraph,
    SessionNode,
    add_edge,
    add_node,
)
from veeksha.generator.interval.registry import IntervalGeneratorRegistry
from veeksha.generator.length.registry import LengthGeneratorRegistry
from veeksha.generator.session_graph.base import BaseSessionGraphGenerator


class LinearSessionGraphGenerator(BaseSessionGraphGenerator):
    def __init__(
        self, config: LinearSessionGraphGeneratorConfig, seed_manager: SeedManager
    ):
        self.config = config
        self.seed_manager = seed_manager
        self.length_rng_factory = seed_manager.numpy_factory("num_request")
        self.interval_rng_factory = seed_manager.numpy_factory("request_wait")
        self.request_wait_generator = IntervalGeneratorRegistry.get(
            config.request_wait_generator.get_type(),
            config.request_wait_generator,
            rng=self.interval_rng_factory(),
        )
        self.num_request_generator = LengthGeneratorRegistry.get(
            config.num_request_generator.get_type(),
            config.num_request_generator,
            rng=self.length_rng_factory(),
        )

    def generate_session_graph(self) -> SessionGraph:
        session_graph = SessionGraph()
        node_id = 0
        num_requests = self.num_request_generator.get_next_value()
        for i in range(num_requests):
            if i == 0:
                wait_time = 0
            else:
                wait_time = self.request_wait_generator.get_next_interval()
            node = SessionNode(id=node_id, wait_after_ready=wait_time)
            add_node(session_graph, node)
            if i > 0:
                edge = SessionEdge(
                    src=node_id - 1,
                    dst=node_id,
                    is_history_parent=self.config.inherit_history,  # all parents are history parents in a linear session
                )
                add_edge(session_graph, edge)
            node_id += 1
        return session_graph
