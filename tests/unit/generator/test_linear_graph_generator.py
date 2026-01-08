import pytest
from veeksha.config.generator.session_graph import LinearSessionGraphGeneratorConfig
from veeksha.config.generator.length import UniformLengthGeneratorConfig
from veeksha.core.seeding import SeedManager
from veeksha.generator.session_graph.linear import LinearSessionGraphGenerator
from veeksha.core.session_graph import SessionGraph


@pytest.fixture
def seed_manager():
    return SeedManager(seed=42)


@pytest.mark.unit
def test_linear_graph_generator_inherit_history_true(seed_manager):
    """Verify that edges have is_history_parent=True when configured."""
    # Configure generator to produce exactly 2 requests
    config = LinearSessionGraphGeneratorConfig(
        num_request_generator=UniformLengthGeneratorConfig(min=2, max=2),
        inherit_history=True
    )
    generator = LinearSessionGraphGenerator(config, seed_manager)
    graph = generator.generate_session_graph()
    
    # Check edges
    # Graph should have 2 nodes (0, 1) and 1 edge (0->1)
    assert len(graph.nodes) == 2
    edges = []
    for edge_list in graph.outgoing.values():
        edges.extend(edge_list)
    assert len(edges) == 1
    
    edge = edges[0]
    assert edge.src == 0
    assert edge.dst == 1
    assert edge.is_history_parent is True


@pytest.mark.unit
def test_linear_graph_generator_inherit_history_false(seed_manager):
    """Verify that edges have is_history_parent=False when configured."""
    # Configure generator to produce exactly 2 requests
    config = LinearSessionGraphGeneratorConfig(
        num_request_generator=UniformLengthGeneratorConfig(min=2, max=2),
        inherit_history=False
    )
    generator = LinearSessionGraphGenerator(config, seed_manager)
    graph = generator.generate_session_graph()
    
    # Check edges
    # Graph should have 2 nodes (0, 1) and 1 edge (0->1)
    assert len(graph.nodes) == 2
    edges = []
    for edge_list in graph.outgoing.values():
        edges.extend(edge_list)
    assert len(edges) == 1
    
    edge = edges[0]
    assert edge.src == 0
    assert edge.dst == 1
    assert edge.is_history_parent is False
