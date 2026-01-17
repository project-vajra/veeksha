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


def is_root(graph: SessionGraph, node_id: int) -> bool:
    return len(parents(graph, node_id)) == 0


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


def render_session_graph(
    graph: SessionGraph,
    output_path: Optional[str] = None,
    format: str = "png",
    view: bool = False,
) -> str:
    """
    Render session graph using Graphviz.

    Generates a proper graph visualization with:
    - Nodes arranged by depth (layers)
    - Edges with arrows and proper routing
    - History parent edges in bold/red
    - Skip connections visible

    Args:
        graph: The SessionGraph to render
        output_path: Path to save the image (without extension).
                     If None, returns DOT source only.
        format: Output format (png, svg, pdf)
        view: If True, open the rendered image

    Returns:
        DOT source string

        >>> from veeksha.core.session_graph import render_session_graph
        >>> render_session_graph(graph, "my_graph", format="png")
        # Creates my_graph.png
    """
    try:
        import graphviz  # type: ignore
    except ImportError:
        return (
            "ERROR: graphviz not installed. Install with:\n"
            "  pip install graphviz\n"
            "And ensure Graphviz is installed on your system:\n"
            "  apt-get install graphviz  # Ubuntu/Debian\n"
            "  brew install graphviz     # macOS"
        )

    # Create directed graph
    dot = graphviz.Digraph(comment="Session Graph")
    dot.attr(rankdir="TB")  # Top to bottom
    dot.attr("node", shape="circle", style="filled", fillcolor="lightblue")

    # Compute depths using LONGEST path from roots (matches generation layers)
    # For DAGs, longest path = topological layer = correct visualization
    depths: Dict[int, int] = {}
    roots = [nid for nid in graph.nodes if len(parents(graph, nid)) == 0]

    # Initialize roots at depth 0
    for r in roots:
        depths[r] = 0

    # Process in topological order, taking max depth from parents + 1
    # Use modified BFS that updates depth to maximum seen
    from collections import deque

    queue = deque(roots)
    while queue:
        node_id = queue.popleft()
        current_depth = depths[node_id]
        for edge in children(graph, node_id):
            child_id = edge.dst
            new_depth = current_depth + 1
            if child_id not in depths or depths[child_id] < new_depth:
                depths[child_id] = new_depth
                queue.append(child_id)

    # Group by depth for subgraph ranking
    max_depth = max(depths.values()) if depths else 0
    levels: Dict[int, List[int]] = {d: [] for d in range(max_depth + 1)}
    for nid, d in depths.items():
        levels[d].append(nid)

    # Add nodes with same rank per level
    for depth in range(max_depth + 1):
        with dot.subgraph() as s:  # type: ignore
            s.attr(rank="same")
            for nid in sorted(levels[depth]):
                node = graph.nodes[nid]
                label = f"{nid}"
                if node.wait_after_ready > 0:
                    label += f"\\n({node.wait_after_ready:.1f}s)"
                s.node(str(nid), label)

    # Add edges with skip connection highlighting
    for src_id in graph.nodes:
        src_depth = depths.get(src_id, 0)
        for edge in children(graph, src_id):
            dst_depth = depths.get(edge.dst, 0)
            layer_span = dst_depth - src_depth

            if edge.is_history_parent:
                # History parent: red, bold
                if layer_span > 1:
                    # Skip connection + history parent: red, bold, dashed
                    dot.edge(
                        str(edge.src),
                        str(edge.dst),
                        color="red",
                        penwidth="2",
                        style="dashed",
                        label=f"H (+{layer_span})",
                    )
                else:
                    dot.edge(
                        str(edge.src),
                        str(edge.dst),
                        color="red",
                        penwidth="2",
                        label="H",
                    )
            else:
                if layer_span > 1:
                    # Skip connection: blue, dashed
                    dot.edge(
                        str(edge.src),
                        str(edge.dst),
                        color="blue",
                        style="dashed",
                        label=f"+{layer_span}",
                    )
                else:
                    # Regular edge: gray
                    dot.edge(str(edge.src), str(edge.dst), color="gray")

    # Render if path provided
    if output_path:
        dot.render(output_path, format=format, view=view, cleanup=True)

    return dot.source
