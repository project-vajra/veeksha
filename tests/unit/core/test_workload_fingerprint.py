"""Unit tests for workload fingerprinting."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from veeksha.core.request import Request
from veeksha.core.request_content import (
    AudioChannelRequestContent,
    TextChannelRequestContent,
)
from veeksha.core.session import Session
from veeksha.core.session_graph import (
    SessionEdge,
    SessionGraph,
    SessionNode,
    add_edge,
    add_node,
)
from veeksha.core.workload_fingerprint import (
    WorkloadFingerprint,
    describe_drift,
    fingerprint_sessions,
    serialize_channel_content,
)
from veeksha.types import ChannelModality


def _session(
    session_id: int = 1,
    text: str = "hello",
    *,
    history: list | None = None,
    dispatch_ticket: int = 0,
    audio_path: str | None = None,
) -> Session:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    channels = {ChannelModality.TEXT: TextChannelRequestContent(input_text=text)}
    if audio_path is not None:
        channels[ChannelModality.AUDIO] = AudioChannelRequestContent(
            input_audio=audio_path
        )
    request = Request(
        id=session_id * 10,
        channels=channels,
        history=list(history or []),
        dispatch_ticket=dispatch_ticket,
        metadata={"k": "v"},
    )
    return Session(id=session_id, session_graph=graph, requests={0: request})


@pytest.mark.unit
def test_same_sessions_same_fingerprint() -> None:
    sessions = [_session(1, "a"), _session(2, "b")]
    assert fingerprint_sessions(sessions) == fingerprint_sessions(
        [_session(1, "a"), _session(2, "b")]
    )


@pytest.mark.unit
def test_different_content_different_fingerprint() -> None:
    assert fingerprint_sessions([_session(1, "a")]) != fingerprint_sessions(
        [_session(1, "b")]
    )


@pytest.mark.unit
def test_history_and_dispatch_ticket_excluded() -> None:
    base = _session(1, "same", history=[], dispatch_ticket=0)
    with_runtime = _session(
        1, "same", history=[{"role": "assistant", "content": "x"}], dispatch_ticket=99
    )
    assert fingerprint_sessions([base]) == fingerprint_sessions([with_runtime])


@pytest.mark.unit
def test_order_matters() -> None:
    a, b = _session(1, "a"), _session(2, "b")
    assert fingerprint_sessions([a, b]) != fingerprint_sessions([b, a])


@pytest.mark.unit
def test_graph_edge_order_invariant() -> None:
    def two_node(flip: bool) -> Session:
        graph = SessionGraph()
        add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
        add_node(graph, SessionNode(id=1, wait_after_ready=0.1))
        if flip:
            add_edge(graph, SessionEdge(src=0, dst=1, is_history_parent=True))
        else:
            # same edge content; insertion via helper is deterministic, but
            # construct with identical semantics
            add_edge(graph, SessionEdge(src=0, dst=1, is_history_parent=True))
        reqs = {
            0: Request(
                id=1,
                channels={
                    ChannelModality.TEXT: TextChannelRequestContent(input_text="a")
                },
            ),
            1: Request(
                id=2,
                channels={
                    ChannelModality.TEXT: TextChannelRequestContent(input_text="b")
                },
            ),
        }
        return Session(id=1, session_graph=graph, requests=reqs)

    assert fingerprint_sessions([two_node(False)]) == fingerprint_sessions(
        [two_node(True)]
    )


@pytest.mark.unit
def test_file_backed_channel_hashes_contents(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF-fake-audio-1")
    first = fingerprint_sessions([_session(1, "t", audio_path=str(audio))])
    audio.write_bytes(b"RIFF-fake-audio-2")
    second = fingerprint_sessions([_session(1, "t", audio_path=str(audio))])
    assert first != second


@pytest.mark.unit
def test_file_backed_channel_detects_rewrite_under_identical_mtime(
    tmp_path: Path,
) -> None:
    """A same-size rewrite must be caught even when mtime does not move.

    ext4/overlayfs hand out a coarse st_mtime_ns, so two quick writes of equal
    length share a timestamp and collide in the digest memo. APFS resolves them,
    which is why this only ever failed on Linux. Forcing the timestamp back
    reproduces it everywhere.
    """
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF-fake-audio-1")
    stamp = audio.stat()
    first = fingerprint_sessions([_session(1, "t", audio_path=str(audio))])

    audio.write_bytes(b"RIFF-fake-audio-2")  # same length, different bytes
    os.utime(audio, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    second = fingerprint_sessions([_session(1, "t", audio_path=str(audio))])

    assert first != second


@pytest.mark.unit
def test_incremental_counts() -> None:
    fp = WorkloadFingerprint()
    fp.add_session(_session(1))
    fp.add_session(_session(2))
    assert fp.session_count == 2
    assert fp.request_count == 2
    assert fp.digest().startswith("blake2b:")
    summary = fp.summary()
    assert summary["sessions"] == 2
    assert summary["workload_fingerprint"] == fp.digest()


@pytest.mark.unit
def test_serialize_channel_content_dataclass() -> None:
    content = TextChannelRequestContent(input_text="hi", target_prompt_tokens=3)
    assert serialize_channel_content(content) == {
        "input_text": "hi",
        "target_prompt_tokens": 3,
    }


@pytest.mark.unit
def test_describe_drift_names_changed_inputs() -> None:
    expected = {
        "veeksha": {"git_commit": "aaa", "version": "1.0"},
        "packages": {"transformers": "4.0"},
        "tokenizer": {"model": "old"},
        "seed": 1,
        "assets": [{"path": "a.wav", "digest": "sha256:1"}],
    }
    actual = {
        "veeksha": {"git_commit": "bbb", "version": "1.0"},
        "packages": {"transformers": "4.1"},
        "tokenizer": {"model": "old"},
        "seed": 1,
        "assets": [{"path": "a.wav", "digest": "sha256:2"}],
    }
    reasons = describe_drift(expected, actual)
    joined = "\n".join(reasons)
    assert "veeksha git commit" in joined
    assert "transformers version" in joined
    assert "asset a.wav" in joined


def test_describe_drift_stays_silent_without_an_actual_record() -> None:
    """Pre-flight has no run record yet; inventing "-> None" diffs misleads.

    Before this, a preflight mismatch blamed the tokenizer and git commit for
    what was really an edited config.
    """
    expected = {
        "veeksha": {"git_commit": "aaa", "version": "1.0"},
        "tokenizer": {"model": "gpt2"},
    }

    assert describe_drift(expected, {}) == []


def test_describe_drift_survives_assets_without_a_path() -> None:
    """A malformed asset must not crash the code explaining a mismatch."""
    expected = {"assets": [{"digest": "sha256:1"}]}
    actual = {"assets": [{"path": "a.wav", "digest": "sha256:2"}]}

    reasons = describe_drift(expected, actual)

    assert any("a.wav" in reason for reason in reasons)
