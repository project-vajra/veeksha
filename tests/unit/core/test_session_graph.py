"""Unit tests for SessionGraph."""

import pytest

from veeksha.core.session_graph import (
    SessionEdge,
    SessionGraph,
    SessionNode,
    add_edge,
    add_node,
    children,
    is_ready,
    is_root,
    parents,
    ready_at,
    topological_order,
)


@pytest.mark.unit
def test_add_node() -> None:
    graph = SessionGraph()
    node = SessionNode(id=1, wait_after_ready=0.1)
    add_node(graph, node)
    assert 1 in graph.nodes
    assert graph.nodes[1] == node

    with pytest.raises(ValueError, match="already exists"):
        add_node(graph, SessionNode(id=1, wait_after_ready=0.2))


@pytest.mark.unit
def test_add_edge() -> None:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=1, wait_after_ready=0))
    add_node(graph, SessionNode(id=2, wait_after_ready=0))

    edge = SessionEdge(src=1, dst=2)
    add_edge(graph, edge)
    
    ps = parents(graph, 2)
    assert len(ps) == 1
    assert ps[0] == edge

    cs = children(graph, 1)
    assert len(cs) == 1
    assert cs[0] == edge

    with pytest.raises(ValueError, match="Both endpoints must exist"):
        add_edge(graph, SessionEdge(src=1, dst=3))


@pytest.mark.unit
def test_ready_at_no_parents() -> None:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=1, wait_after_ready=0.5))
    
    # Should be ready immediately + delay
    assert ready_at(graph, 1, {}) == 0.5


@pytest.mark.unit
def test_ready_at_with_dependencies() -> None:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=1, wait_after_ready=0))
    add_node(graph, SessionNode(id=2, wait_after_ready=0.1))
    add_edge(graph, SessionEdge(src=1, dst=2))

    # Parent not complete -> None
    assert ready_at(graph, 2, {}) is None
    
    # Parent complete at 10.0 -> Ready at 10.0 + 0.1
    assert ready_at(graph, 2, {1: 10.0}) == 10.1


@pytest.mark.unit
def test_ready_at_multiple_parents() -> None:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=1, wait_after_ready=0))
    add_node(graph, SessionNode(id=2, wait_after_ready=0))
    add_node(graph, SessionNode(id=3, wait_after_ready=0.2))
    
    add_edge(graph, SessionEdge(src=1, dst=3))
    add_edge(graph, SessionEdge(src=2, dst=3))

    # Only one parent done
    assert ready_at(graph, 3, {1: 5.0}) is None
    
    # Both done, max time used
    assert ready_at(graph, 3, {1: 5.0, 2: 6.0}) == 6.2


@pytest.mark.unit
def test_is_ready() -> None:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=1, wait_after_ready=1.0))
    
    # ready_at returns 1.0. Current time 0.5 -> False
    assert not is_ready(graph, 1, {}, now=0.5)
    
    # Current time 1.0 -> True
    assert is_ready(graph, 1, {}, now=1.0)
    
    # Current time 2.0 -> True
    assert is_ready(graph, 1, {}, now=2.0)


@pytest.mark.unit
def test_topological_order() -> None:
    graph = SessionGraph()
    # 1 -> 2 -> 3
    # 1 -> 4
    add_node(graph, SessionNode(id=1, wait_after_ready=0))
    add_node(graph, SessionNode(id=2, wait_after_ready=0))
    add_node(graph, SessionNode(id=3, wait_after_ready=0))
    add_node(graph, SessionNode(id=4, wait_after_ready=0))
    
    add_edge(graph, SessionEdge(src=1, dst=2))
    add_edge(graph, SessionEdge(src=2, dst=3))
    add_edge(graph, SessionEdge(src=1, dst=4))

    order = topological_order(graph)
    assert len(order) == 4
    assert order.index(1) < order.index(2)
    assert order.index(2) < order.index(3)
    assert order.index(1) < order.index(4)


@pytest.mark.unit
def test_topological_order_cycle() -> None:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=1, wait_after_ready=0))
    add_node(graph, SessionNode(id=2, wait_after_ready=0))
    
    add_edge(graph, SessionEdge(src=1, dst=2))
    add_edge(graph, SessionEdge(src=2, dst=1))

    with pytest.raises(ValueError, match="cycle"):
        topological_order(graph)


@pytest.mark.unit
def test_is_root() -> None:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=1, wait_after_ready=0))
    add_node(graph, SessionNode(id=2, wait_after_ready=0))
    add_edge(graph, SessionEdge(src=1, dst=2))

    assert is_root(graph, 1)
    assert not is_root(graph, 2)
