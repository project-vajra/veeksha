from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SessionNode:
    id: int
    wait_after_ready: float


@dataclass
class SessionEdge:
    src: int
    dst: int
    is_history_parent: bool = True


@dataclass
class SessionGraph:
    nodes: Dict[int, SessionNode] = field(default_factory=dict)
    outgoing: Dict[int, List[SessionEdge]] = field(
        default_factory=dict
    )  # outgoing edges from a node
    incoming: Dict[int, List[SessionEdge]] = field(
        default_factory=dict
    )  # incoming edges to a node


def add_node(graph: SessionGraph, node: SessionNode) -> None:
    if node.id in graph.nodes:
        raise ValueError(f"Node {node.id} already exists")
    graph.nodes[node.id] = node


def add_edge(graph: SessionGraph, edge: SessionEdge) -> None:
    if edge.src not in graph.nodes or edge.dst not in graph.nodes:
        raise ValueError("Both endpoints must exist before adding an edge")
    graph.outgoing.setdefault(edge.src, []).append(edge)
    graph.incoming.setdefault(edge.dst, []).append(edge)


def parents(graph: SessionGraph, node_id: int) -> List[SessionEdge]:
    return graph.incoming.get(node_id, [])


def children(graph: SessionGraph, node_id: int) -> List[SessionEdge]:
    return graph.outgoing.get(node_id, [])


def get_node_ids(graph: SessionGraph) -> List[int]:
    return list(graph.nodes.keys())


def ready_at(
    graph: SessionGraph, node_id: int, completions: Dict[int, float]
) -> Optional[float]:
    """
    Returns the time at which the node is ready to be executed.

    If a parent completion time is missing, return None
    """
    ps = parents(graph, node_id)
    if not ps:
        return graph.nodes[node_id].wait_after_ready
    parent_times = []
    for edge in ps:
        if edge.src not in completions:
            return None
        parent_times.append(completions[edge.src])
    parent_finish = max(parent_times)
    return parent_finish + graph.nodes[node_id].wait_after_ready


def is_ready(
    graph: SessionGraph, node_id: int, completions: Dict[int, float], now: float
) -> bool:
    ready_time = ready_at(graph, node_id, completions)
    if ready_time is None:
        return False
    return now >= ready_time


def topological_order(graph: SessionGraph) -> List[int]:
    incoming_counts = {nid: len(parents(graph, nid)) for nid in graph.nodes}
    queue = [nid for nid, deg in incoming_counts.items() if deg == 0]
    order: List[int] = []
    while queue:
        current = queue.pop()
        order.append(current)
        for edge in children(graph, current):
            incoming_counts[edge.dst] -= 1
            if incoming_counts[edge.dst] == 0:
                queue.append(edge.dst)
    if len(order) != len(graph.nodes):
        raise ValueError("SessionGraph contains a cycle")
    return order


def format_session_graph(graph: SessionGraph) -> str:
    lines = []
    lines.append("  Nodes:")
    for node_id, node in sorted(graph.nodes.items()):
        lines.append(f"    {node_id} -> wait_after_ready={node.wait_after_ready}")
    lines.append("  Edges:")
    seen_edges = []
    for edges in graph.outgoing.values():
        seen_edges.extend(edges)
    for edge in sorted(seen_edges, key=lambda e: (e.src, e.dst)):
        lines.append(f"    {edge.src} -> {edge.dst}")
    return "\n".join(lines)


def print_session_graph(graph: SessionGraph) -> None:
    print(format_session_graph(graph))


if __name__ == "__main__":
    graph = SessionGraph()
    add_node(graph, SessionNode(id=1, wait_after_ready=0))
    add_node(graph, SessionNode(id=2, wait_after_ready=0))
    add_edge(graph, SessionEdge(src=1, dst=2))
    print(f"parents of 2: {parents(graph, 2)}")
    print(f"children of 1: {children(graph, 1)}")
    print(f"ready_at of 2: {ready_at(graph, 2, {})}")
    print(f"is_ready: {is_ready(graph, 2, {1: 0}, 0)}")
    try:
        print(f"topological_order: {topological_order(graph)}")
    except ValueError as e:
        print(f"Error: {e}")
    print_session_graph(graph)
