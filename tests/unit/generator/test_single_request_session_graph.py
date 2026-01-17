"""Tests for SingleRequestSessionGraphGenerator."""

import pytest

from veeksha.config.generator.session_graph import SingleRequestSessionGraphGeneratorConfig
from veeksha.core.seeding import SeedManager
from veeksha.core.session_graph import parents, topological_order
from veeksha.generator.session_graph.single_request import SingleRequestSessionGraphGenerator


class TestSingleRequestSessionGraphGenerator:
    """Tests for SingleRequestSessionGraphGenerator."""

    def test_single_node(self):
        """Should generate exactly one node."""
        config = SingleRequestSessionGraphGeneratorConfig()
        seed_manager = SeedManager(42)
        generator = SingleRequestSessionGraphGenerator(config, seed_manager)
        graph = generator.generate_session_graph()

        assert len(graph.nodes) == 1
        assert 0 in graph.nodes

    def test_no_edges(self):
        """Single node graph should have no edges."""
        config = SingleRequestSessionGraphGeneratorConfig()
        seed_manager = SeedManager(42)
        generator = SingleRequestSessionGraphGenerator(config, seed_manager)
        graph = generator.generate_session_graph()

        assert len(graph.outgoing) == 0
        assert len(graph.incoming) == 0

    def test_node_is_root(self):
        """The single node should be a root (no parents)."""
        config = SingleRequestSessionGraphGeneratorConfig()
        seed_manager = SeedManager(42)
        generator = SingleRequestSessionGraphGenerator(config, seed_manager)
        graph = generator.generate_session_graph()

        incoming = parents(graph, 0)
        assert len(incoming) == 0

    def test_zero_wait_time(self):
        """The single node should have zero wait time."""
        config = SingleRequestSessionGraphGeneratorConfig()
        seed_manager = SeedManager(42)
        generator = SingleRequestSessionGraphGenerator(config, seed_manager)
        graph = generator.generate_session_graph()

        assert graph.nodes[0].wait_after_ready == 0

    def test_valid_dag(self):
        """Single node is a valid DAG."""
        config = SingleRequestSessionGraphGeneratorConfig()
        seed_manager = SeedManager(42)
        generator = SingleRequestSessionGraphGenerator(config, seed_manager)
        graph = generator.generate_session_graph()

        order = topological_order(graph)
        assert order == [0]

    def test_reproducible(self):
        """Same seed should produce identical graphs."""
        config = SingleRequestSessionGraphGeneratorConfig()

        gen1 = SingleRequestSessionGraphGenerator(config, SeedManager(123))
        gen2 = SingleRequestSessionGraphGenerator(config, SeedManager(123))

        graph1 = gen1.generate_session_graph()
        graph2 = gen2.generate_session_graph()

        assert len(graph1.nodes) == len(graph2.nodes)
        assert list(graph1.nodes.keys()) == list(graph2.nodes.keys())
