from typing import List, Set, Tuple

from veeksha.config.generator.session_graph import BranchingSessionGraphGeneratorConfig
from veeksha.core.seeding import SeedManager
from veeksha.core.session_graph import (
    SessionEdge,
    SessionGraph,
    SessionNode,
    add_edge,
    add_node,
    parents,
)
from veeksha.generator.interval.registry import IntervalGeneratorRegistry
from veeksha.generator.length.registry import LengthGeneratorRegistry
from veeksha.generator.session_graph.base import BaseSessionGraphGenerator


class BranchingSessionGraphGenerator(BaseSessionGraphGenerator):
    """
    Generates branching session graphs (DAGs) with configurable:
    - Number of layers (depth)
    - Layer width (per-layer)
    - Fan-out (forward connections)
    - Fan-in (backward connections / minimum parents)
    - Connection distance (skip layers)
    - History inheritance (one parent per node)
    """

    def __init__(
        self, config: BranchingSessionGraphGeneratorConfig, seed_manager: SeedManager
    ):
        self.config = config
        self.seed_manager = seed_manager

        # Create RNG factories for reproducibility
        self.num_layers_rng = seed_manager.numpy_factory("num_layers")()
        self.layer_width_rng = seed_manager.numpy_factory("layer_width")()
        self.fan_out_rng = seed_manager.numpy_factory("fan_out")()
        self.fan_in_rng = seed_manager.numpy_factory("fan_in")()
        self.connection_dist_rng = seed_manager.numpy_factory("connection_dist")()
        self.interval_rng = seed_manager.numpy_factory("request_wait")()
        self.history_rng = seed_manager.numpy_factory("history")()

        # Instantiate generators
        self.num_layers_gen = LengthGeneratorRegistry.get(
            config.num_layers_generator.get_type(),
            config.num_layers_generator,
            rng=self.num_layers_rng,
        )
        self.layer_width_gen = LengthGeneratorRegistry.get(
            config.layer_width_generator.get_type(),
            config.layer_width_generator,
            rng=self.layer_width_rng,
        )
        self.fan_out_gen = LengthGeneratorRegistry.get(
            config.fan_out_generator.get_type(),
            config.fan_out_generator,
            rng=self.fan_out_rng,
        )
        self.fan_in_gen = LengthGeneratorRegistry.get(
            config.fan_in_generator.get_type(),
            config.fan_in_generator,
            rng=self.fan_in_rng,
        )
        self.connection_dist_gen = LengthGeneratorRegistry.get(
            config.connection_dist_generator.get_type(),
            config.connection_dist_generator,
            rng=self.connection_dist_rng,
        )
        self.request_wait_gen = IntervalGeneratorRegistry.get(
            config.request_wait_generator.get_type(),
            config.request_wait_generator,
            rng=self.interval_rng,
        )

    def generate_session_graph(self) -> SessionGraph:
        graph = SessionGraph()

        # Step 1: Generate layers
        num_layers = self.num_layers_gen.get_next_value()
        layers: List[List[int]] = []
        node_id = 0

        for layer_idx in range(num_layers):
            if layer_idx == 0 and self.config.single_root:
                width = 1
            else:
                width = self.layer_width_gen.get_next_value()

            layer_nodes = []
            for _ in range(width):
                wait_time = (
                    0 if layer_idx == 0 else self.request_wait_gen.get_next_interval()
                )
                node = SessionNode(id=node_id, wait_after_ready=wait_time)
                add_node(graph, node)
                layer_nodes.append(node_id)
                node_id += 1
            layers.append(layer_nodes)

        # Step 2: Forward pass (fan-out with skip connections)
        edges_to_add: Set[Tuple[int, int]] = set()

        for layer_idx in range(num_layers - 1):
            for u in layers[layer_idx]:
                num_children = self.fan_out_gen.get_next_value()
                for _ in range(num_children):
                    skip = self.connection_dist_gen.get_next_value()
                    target_layer = min(layer_idx + skip, num_layers - 1)
                    if target_layer > layer_idx and layers[target_layer]:
                        v = int(self.history_rng.choice(layers[target_layer]))
                        edges_to_add.add((u, v))

        # Step 3: Backward pass (fan-in / ensure connectivity)
        for layer_idx in range(1, num_layers):
            for v in layers[layer_idx]:
                # Count existing parents from edges_to_add
                current_parents = sum(1 for (src, dst) in edges_to_add if dst == v)
                sampled_fan_in = self.fan_in_gen.get_next_value()
                # Cap needed to available nodes in previous layer
                max_possible = len(layers[layer_idx - 1])
                needed = min(max(1, sampled_fan_in), max_possible)

                while current_parents < needed:
                    u = int(self.history_rng.choice(layers[layer_idx - 1]))
                    if (u, v) not in edges_to_add:
                        edges_to_add.add((u, v))
                        current_parents += 1

        # Step 4: Add edges to graph (initially all non-history)
        for src, dst in edges_to_add:
            edge = SessionEdge(src=src, dst=dst, is_history_parent=False)
            add_edge(graph, edge)

        # Step 5: History inheritance
        if self.config.inherit_history:
            self._assign_history_parents(graph)

        return graph

    def _assign_history_parents(self, graph: SessionGraph) -> None:
        """For each node with parents, randomly select exactly one as history parent."""
        for node_id in graph.nodes:
            incoming = parents(graph, node_id)
            if not incoming:
                continue

            # Select one random edge to be the history parent
            chosen_idx = self.history_rng.randint(0, len(incoming))
            for idx, edge in enumerate(incoming):
                edge.is_history_parent = idx == chosen_idx
