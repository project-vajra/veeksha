"""Tests for the timed synthetic session trace flavor generator."""

import json
from unittest.mock import MagicMock

import pytest

from veeksha.config.generator.session import (
    TimedSyntheticSessionTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.core.session_graph import parents
from veeksha.generator.session.trace.timed_synthetic_session import (
    TimedSyntheticSessionTraceFlavorGenerator,
)
from veeksha.types import ChannelModality


@pytest.fixture
def mock_tokenizer_provider():
    provider = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = lambda text: text.split() if text else []
    tokenizer.decode = (
        lambda token_ids, skip_special_tokens=False: " ".join(str(t) for t in token_ids)
    )
    tokenizer.count_tokens = lambda text: len(text.split()) if text else 0
    tokenizer.get_vocab.return_value = list(range(2048))
    provider.for_modality.return_value = tokenizer
    return provider


@pytest.fixture
def seed_manager():
    return SeedManager(seed=42)


def _write_trace(tmp_path, rows):
    path = tmp_path / "trace.jsonl"
    with open(path, "w") as fd:
        for row in rows:
            fd.write(json.dumps(row) + "\n")
    return str(path)


def _make_generator(trace_file, seed_manager, mock_tokenizer_provider, wrap_mode=False):
    flavor_config = TimedSyntheticSessionTraceFlavorConfig()
    config = TraceSessionGeneratorConfig(
        trace_file=trace_file,
        flavor=flavor_config,
        wrap_mode=wrap_mode,
    )
    return TimedSyntheticSessionTraceFlavorGenerator(
        config,
        flavor_config,
        seed_manager,
        mock_tokenizer_provider,
    )


def test_legacy_linear_trace_falls_back_to_row_order(
    tmp_path, seed_manager, mock_tokenizer_provider
):
    trace_file = _write_trace(
        tmp_path,
        [
            {
                "session_id": 3,
                "input_length": 12,
                "new_input_length": 12,
                "output_length": 4,
                "wait_after_previous_response_s": 0.0,
            },
            {
                "session_id": 3,
                "input_length": 18,
                "new_input_length": 6,
                "output_length": 7,
                "wait_after_previous_response_s": 0.25,
            },
        ],
    )
    gen = _make_generator(trace_file, seed_manager, mock_tokenizer_provider)

    session = gen.generate_session()
    assert set(session.requests.keys()) == {0, 1}

    incoming = parents(session.session_graph, 1)
    assert len(incoming) == 1
    assert incoming[0].src == 0
    assert incoming[0].is_history_parent is True

    ctx = session.requests[1].session_context
    assert ctx["node_id"] == 1
    assert ctx["parent_nodes"] == [0]
    assert ctx["history_parent"] == 0
    assert ctx["wait_after_ready"] == pytest.approx(0.25)


def test_replays_dag_topology_from_session_context(
    tmp_path, seed_manager, mock_tokenizer_provider
):
    trace_file = _write_trace(
        tmp_path,
        [
            {
                "session_id": 8,
                "input_length": 8,
                "new_input_length": 8,
                "output_length": 4,
                "session_context": {
                    "node_id": 1,
                    "parent_nodes": [],
                    "history_parent": None,
                    "wait_after_ready": 0.1,
                },
            },
            {
                "session_id": 8,
                "input_length": 8,
                "new_input_length": 8,
                "output_length": 4,
                "session_context": {
                    "node_id": 0,
                    "parent_nodes": [],
                    "history_parent": None,
                    "wait_after_ready": 0.0,
                },
            },
            {
                "session_id": 8,
                "input_length": 16,
                "new_input_length": 8,
                "output_length": 5,
                "session_context": {
                    "node_id": 2,
                    "parent_nodes": [0, 1],
                    "history_parent": 1,
                    "wait_after_ready": 0.2,
                },
            },
        ],
    )
    gen = _make_generator(trace_file, seed_manager, mock_tokenizer_provider)

    session = gen.generate_session()
    assert set(session.requests.keys()) == {0, 1, 2}

    incoming = sorted(
        (edge.src, edge.is_history_parent) for edge in parents(session.session_graph, 2)
    )
    assert incoming == [(0, False), (1, True)]

    req2 = session.requests[2]
    assert req2.session_context["parent_nodes"] == [0, 1]
    assert req2.session_context["history_parent"] == 1
    assert req2.session_context["wait_after_ready"] == pytest.approx(0.2)

    text0 = session.requests[0].channels[ChannelModality.TEXT].input_text
    text1 = session.requests[1].channels[ChannelModality.TEXT].input_text
    text2 = req2.channels[ChannelModality.TEXT].input_text
    assert text1 == text2
    assert text0 != text1


def test_dependency_without_history_parent_starts_new_lineage(
    tmp_path, seed_manager, mock_tokenizer_provider
):
    trace_file = _write_trace(
        tmp_path,
        [
            {
                "session_id": 9,
                "input_length": 8,
                "new_input_length": 8,
                "output_length": 4,
                "session_context": {
                    "node_id": 0,
                    "parent_nodes": [],
                    "history_parent": None,
                    "wait_after_ready": 0.0,
                },
            },
            {
                "session_id": 9,
                "input_length": 8,
                "new_input_length": 8,
                "output_length": 4,
                "session_context": {
                    "node_id": 1,
                    "parent_nodes": [],
                    "history_parent": None,
                    "wait_after_ready": 0.0,
                },
            },
            {
                "session_id": 9,
                "input_length": 16,
                "new_input_length": 8,
                "output_length": 5,
                "session_context": {
                    "node_id": 2,
                    "parent_nodes": [0, 1],
                    "history_parent": None,
                    "wait_after_ready": 0.0,
                },
            },
        ],
    )
    gen = _make_generator(trace_file, seed_manager, mock_tokenizer_provider)

    session = gen.generate_session()
    prompts = {
        node_id: request.channels[ChannelModality.TEXT].input_text
        for node_id, request in session.requests.items()
    }
    assert prompts[2] != prompts[0]
    assert prompts[2] != prompts[1]


def test_wrap_regenerates_prompts_for_entire_history_lineage(
    tmp_path, seed_manager, mock_tokenizer_provider
):
    trace_file = _write_trace(
        tmp_path,
        [
            {
                "session_id": 1,
                "input_length": 8,
                "new_input_length": 8,
                "output_length": 4,
                "session_context": {
                    "node_id": 0,
                    "parent_nodes": [],
                    "history_parent": None,
                    "wait_after_ready": 0.0,
                },
            },
            {
                "session_id": 1,
                "input_length": 16,
                "new_input_length": 8,
                "output_length": 5,
                "session_context": {
                    "node_id": 1,
                    "parent_nodes": [0],
                    "history_parent": 0,
                    "wait_after_ready": 0.1,
                },
            },
        ],
    )
    gen = _make_generator(
        trace_file,
        seed_manager,
        mock_tokenizer_provider,
        wrap_mode=True,
    )

    session_epoch_1 = gen.generate_session()
    session_epoch_2 = gen.generate_session()

    epoch_1_root = session_epoch_1.requests[0].channels[ChannelModality.TEXT].input_text
    epoch_1_child = session_epoch_1.requests[1].channels[ChannelModality.TEXT].input_text
    epoch_2_root = session_epoch_2.requests[0].channels[ChannelModality.TEXT].input_text
    epoch_2_child = session_epoch_2.requests[1].channels[ChannelModality.TEXT].input_text

    assert epoch_1_root == epoch_1_child
    assert epoch_2_root == epoch_2_child
    assert epoch_1_root != epoch_2_root
    assert epoch_1_child != epoch_2_child
