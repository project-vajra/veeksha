from dataclasses import dataclass
from typing import Dict

from veeksha.core.request import Request
from veeksha.core.session_graph import SessionGraph, format_session_graph


@dataclass
class Session:
    """A single Veeksha session.

    Args:
        id: Unique session ID
        session_graph: Session graph of the session (just structure, no content)
        requests: Requests in the session (actual content) indexed by node id of the session graph
    """

    id: int
    session_graph: SessionGraph
    requests: Dict[int, Request]


def format_session(session: Session) -> str:
    """Return a human-readable representation of a session."""

    lines = [
        f"Session {session.id}:",
        "  Graph:",
    ]
    graph_lines = format_session_graph(session.session_graph).split("\n")
    lines.extend(f"  {line}" for line in graph_lines)
    lines.append("  Requests:")
    for request_id, request in sorted(session.requests.items()):
        channel_names = ", ".join(str(modality) for modality in request.channels)
        lines.append(
            "    " f"{request_id} -> id={request.id}, channels=[{channel_names}]"
        )
    return "\n".join(lines)


def print_session(session: Session) -> None:
    """Print a human-readable representation of a session."""

    print(format_session(session))
