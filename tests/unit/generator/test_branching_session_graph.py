"""Tests for BranchingSessionGraphGenerator."""

import pytest

from veeksha.config.generator.length import FixedLengthGeneratorConfig
from veeksha.config.generator.session_graph import BranchingSessionGraphGeneratorConfig
from veeksha.core.seeding import SeedManager
from veeksha.core.session_graph import parents, topological_order
from veeksha.generator.session_graph.branching import BranchingSessionGraphGenerator


class TestBranchingSessionGraphGenerator:
    """Tests for BranchingSessionGraphGenerator."""

    def test_single_root_enforced(self):
        """When single_root=True, layer 0 should have exactly 1 node."""
        config = BranchingSessionGraphGeneratorConfig(
            num_layers_generator=FixedLengthGeneratorConfig(value=3),
            layer_width_generator=FixedLengthGeneratorConfig(value=5),
            single_root=True,
        )
        seed_manager = SeedManager(42)
        generator = BranchingSessionGraphGenerator(config, seed_manager)
        graph = generator.generate_session_graph()

        # Node 0 should be the only root
        roots = [nid for nid in graph.nodes if not parents(graph, nid)]
        assert len(roots) == 1
        assert roots[0] == 0

    def test_multiple_roots_allowed(self):
        """When single_root=False, layer 0 can have multiple nodes."""
        config = BranchingSessionGraphGeneratorConfig(
            num_layers_generator=FixedLengthGeneratorConfig(value=2),
            layer_width_generator=FixedLengthGeneratorConfig(value=3),
            single_root=False,
        )
        seed_manager = SeedManager(42)
        generator = BranchingSessionGraphGenerator(config, seed_manager)
        graph = generator.generate_session_graph()

        roots = [nid for nid in graph.nodes if not parents(graph, nid)]
        assert len(roots) == 3  # All 3 nodes in layer 0 are roots

    def test_valid_dag_no_cycles(self):
        """Generated graph should be a valid DAG (no cycles)."""
        config = BranchingSessionGraphGeneratorConfig(
            num_layers_generator=FixedLengthGeneratorConfig(value=4),
            layer_width_generator=FixedLengthGeneratorConfig(value=3),
            fan_out_generator=FixedLengthGeneratorConfig(value=3),
        )
        seed_manager = SeedManager(123)
        generator = BranchingSessionGraphGenerator(config, seed_manager)
        graph = generator.generate_session_graph()

        # Should not raise ValueError
        order = topological_order(graph)
        assert len(order) == len(graph.nodes)

    def test_history_inheritance_one_parent(self):
        """When inherit_history=True, each non-root node has exactly 1 history parent."""
        config = BranchingSessionGraphGeneratorConfig(
            num_layers_generator=FixedLengthGeneratorConfig(value=3),
            layer_width_generator=FixedLengthGeneratorConfig(value=3),
            fan_out_generator=FixedLengthGeneratorConfig(value=3),
            inherit_history=True,
        )
        seed_manager = SeedManager(42)
        generator = BranchingSessionGraphGenerator(config, seed_manager)
        graph = generator.generate_session_graph()

        for node_id in graph.nodes:
            incoming = parents(graph, node_id)
            if not incoming:
                continue  # Root nodes
            history_parents = [e for e in incoming if e.is_history_parent]
            assert len(history_parents) == 1, f"Node {node_id} has {len(history_parents)} history parents"

    def test_no_history_inheritance(self):
        """When inherit_history=False, no edges should be history parents."""
        config = BranchingSessionGraphGeneratorConfig(
            num_layers_generator=FixedLengthGeneratorConfig(value=3),
            layer_width_generator=FixedLengthGeneratorConfig(value=2),
            inherit_history=False,
        )
        seed_manager = SeedManager(42)
        generator = BranchingSessionGraphGenerator(config, seed_manager)
        graph = generator.generate_session_graph()

        for node_id in graph.nodes:
            incoming = parents(graph, node_id)
            for edge in incoming:
                assert not edge.is_history_parent, f"Edge {edge.src}->{edge.dst} should not be history parent"

    def test_connectivity_all_non_roots_have_parents(self):
        """Every non-root node should have at least 1 parent."""
        config = BranchingSessionGraphGeneratorConfig(
            num_layers_generator=FixedLengthGeneratorConfig(value=4),
            layer_width_generator=FixedLengthGeneratorConfig(value=4),
            fan_out_generator=FixedLengthGeneratorConfig(value=1),
            fan_in_generator=FixedLengthGeneratorConfig(value=1),
            single_root=True,
        )
        seed_manager = SeedManager(99)
        generator = BranchingSessionGraphGenerator(config, seed_manager)
        graph = generator.generate_session_graph()

        for node_id in graph.nodes:
            if node_id == 0:  # Root
                continue
            incoming = parents(graph, node_id)
            assert len(incoming) >= 1, f"Node {node_id} has no parents"
