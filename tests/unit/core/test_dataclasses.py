"""Unit tests for Core Dataclasses."""

import pytest

from veeksha.core.request import Request
from veeksha.core.session import Session, format_session
from veeksha.core.session_graph import SessionGraph, SessionNode, add_node
from veeksha.types import ChannelModality


@pytest.mark.unit
def test_request_init() -> None:
    dict_channels = {ChannelModality.TEXT: "something"}
    r = Request(id=1, channels=dict_channels)
    assert r.id == 1
    assert r.channels == dict_channels
    assert str(r) == "RequestConfig(id=1)"


@pytest.mark.unit
def test_session_init_and_format() -> None:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=1, wait_after_ready=0.1))
    
    req = Request(id=101, channels={ChannelModality.TEXT: "content"})
    session = Session(id=5, session_graph=graph, requests={1: req})
    
    assert session.id == 5
    assert len(session.requests) == 1
    
    formatted = format_session(session)
    assert "Session 5:" in formatted
    assert "Nodes:" in formatted
    assert "1 -> wait_after_ready=0.1" in formatted
    assert "Requests:" in formatted
    assert "1 -> id=101, channels=[text]" in formatted
