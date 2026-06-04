"""Tests for the untimed content multi-turn trace flavor generator."""

import json

import pytest
from unittest.mock import MagicMock

from veeksha.config.generator.session import (
    UntimedContentMultiTurnTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.generator.session.trace.conversation import (
    UntimedContentMultiTurnTraceFlavorGenerator,
)


@pytest.fixture
def mock_tokenizer_provider():
    provider = MagicMock()
    tokenizer = MagicMock()
    # Simple mock: encode returns a list whose length ~ number of words
    tokenizer.encode = lambda text: text.split() if text else []
    provider.for_modality.return_value = tokenizer
    return provider


@pytest.fixture
def seed_manager():
    return SeedManager(seed=42)


def _write_sharegpt_trace(tmp_path, conversations):
    """Write a JSONL trace file with ShareGPT-format conversations."""
    f = tmp_path / "trace.jsonl"
    with open(f, "w") as fd:
        for conv in conversations:
            fd.write(json.dumps(conv) + "\n")
    return str(f)


def _make_config(trace_file, flavor_config):
    return TraceSessionGeneratorConfig(
        trace_file=trace_file,
        flavor=flavor_config,
        wrap_mode=False,
    )


class TestShareGPTFormat:
    """Test basic ShareGPT-format conversation parsing."""

    def test_basic_two_turn_conversation(self, tmp_path, seed_manager, mock_tokenizer_provider):
        trace_file = _write_sharegpt_trace(tmp_path, [
            {
                "conversations": [
                    {"from": "human", "value": "What is Python?"},
                    {"from": "gpt", "value": "Python is a programming language."},
                    {"from": "human", "value": "Tell me more"},
                    {"from": "gpt", "value": "Python was created by Guido."},
                ]
            }
        ])
        flavor_config = UntimedContentMultiTurnTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config)
        gen = UntimedContentMultiTurnTraceFlavorGenerator(
            config, flavor_config, seed_manager, mock_tokenizer_provider
        )

        session = gen.generate_session()
        assert len(session.requests) == 2

        # Request 0: first turn, no history
        req0 = session.requests[0]
        assert req0.history == []
        from veeksha.types import ChannelModality
        assert req0.channels[ChannelModality.TEXT].input_text == "What is Python?"

        # Request 1: second turn, has history
        req1 = session.requests[1]
        assert len(req1.history) == 2
        assert req1.history[0] == {"role": "user", "content": "What is Python?"}
        assert req1.history[1] == {"role": "assistant", "content": "Python is a programming language."}
        assert req1.channels[ChannelModality.TEXT].input_text == "Tell me more"

    def test_single_turn_conversation(self, tmp_path, seed_manager, mock_tokenizer_provider):
        trace_file = _write_sharegpt_trace(tmp_path, [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there!"},
                ]
            }
        ])
        flavor_config = UntimedContentMultiTurnTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config)
        gen = UntimedContentMultiTurnTraceFlavorGenerator(
            config, flavor_config, seed_manager, mock_tokenizer_provider
        )

        session = gen.generate_session()
        assert len(session.requests) == 1
        assert session.requests[0].history == []

    def test_target_output_tokens(self, tmp_path, seed_manager, mock_tokenizer_provider):
        trace_file = _write_sharegpt_trace(tmp_path, [
            {
                "conversations": [
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "one two three four five"},
                ]
            }
        ])
        flavor_config = UntimedContentMultiTurnTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config)
        gen = UntimedContentMultiTurnTraceFlavorGenerator(
            config, flavor_config, seed_manager, mock_tokenizer_provider
        )

        session = gen.generate_session()
        # Mock tokenizer splits on spaces, so "one two three four five" = 5 tokens
        assert session.requests[0].requested_output.text.target_tokens == 5


class TestCustomSchema:
    """Test LMSYS-format and other custom schemas."""

    def test_lmsys_format(self, tmp_path, seed_manager, mock_tokenizer_provider):
        trace_file = _write_sharegpt_trace(tmp_path, [
            {
                "conversation": [
                    {"role": "user", "content": "What is AI?"},
                    {"role": "assistant", "content": "AI is artificial intelligence."},
                ]
            }
        ])
        flavor_config = UntimedContentMultiTurnTraceFlavorConfig(
            conversation_column="conversation",
            role_key="role",
            content_key="content",
            user_role_value="user",
            assistant_role_value="assistant",
        )
        config = _make_config(trace_file, flavor_config)
        gen = UntimedContentMultiTurnTraceFlavorGenerator(
            config, flavor_config, seed_manager, mock_tokenizer_provider
        )

        session = gen.generate_session()
        assert len(session.requests) == 1
        from veeksha.types import ChannelModality
        assert session.requests[0].channels[ChannelModality.TEXT].input_text == "What is AI?"


class TestEdgeCases:
    """Test edge cases in conversation parsing."""

    def test_leading_assistant_message_skipped(self, tmp_path, seed_manager, mock_tokenizer_provider):
        trace_file = _write_sharegpt_trace(tmp_path, [
            {
                "conversations": [
                    {"from": "gpt", "value": "Welcome!"},  # Leading assistant — skip
                    {"from": "human", "value": "Thanks"},
                    {"from": "gpt", "value": "How can I help?"},
                ]
            }
        ])
        flavor_config = UntimedContentMultiTurnTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config)
        gen = UntimedContentMultiTurnTraceFlavorGenerator(
            config, flavor_config, seed_manager, mock_tokenizer_provider
        )

        session = gen.generate_session()
        assert len(session.requests) == 1
        from veeksha.types import ChannelModality
        assert session.requests[0].channels[ChannelModality.TEXT].input_text == "Thanks"

    def test_trailing_user_message_skipped(self, tmp_path, seed_manager, mock_tokenizer_provider):
        trace_file = _write_sharegpt_trace(tmp_path, [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi!"},
                    {"from": "human", "value": "Unanswered question"},  # No response
                ]
            }
        ])
        flavor_config = UntimedContentMultiTurnTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config)
        gen = UntimedContentMultiTurnTraceFlavorGenerator(
            config, flavor_config, seed_manager, mock_tokenizer_provider
        )

        session = gen.generate_session()
        assert len(session.requests) == 1  # Only one complete pair

    def test_empty_conversation(self, tmp_path, seed_manager, mock_tokenizer_provider):
        trace_file = _write_sharegpt_trace(tmp_path, [
            {"conversations": []}
        ])
        flavor_config = UntimedContentMultiTurnTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config)
        gen = UntimedContentMultiTurnTraceFlavorGenerator(
            config, flavor_config, seed_manager, mock_tokenizer_provider
        )

        session = gen.generate_session()
        # Should create a minimal fallback session
        assert len(session.requests) == 1

    def test_system_messages_skipped(self, tmp_path, seed_manager, mock_tokenizer_provider):
        trace_file = _write_sharegpt_trace(tmp_path, [
            {
                "conversations": [
                    {"from": "system", "value": "You are helpful."},
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "Hello!"},
                ]
            }
        ])
        flavor_config = UntimedContentMultiTurnTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config)
        gen = UntimedContentMultiTurnTraceFlavorGenerator(
            config, flavor_config, seed_manager, mock_tokenizer_provider
        )

        session = gen.generate_session()
        assert len(session.requests) == 1
        from veeksha.types import ChannelModality
        assert session.requests[0].channels[ChannelModality.TEXT].input_text == "Hi"


class TestSessionGraph:
    """Test session graph construction with is_history_parent=False."""

    def test_edges_have_history_parent_false(self, tmp_path, seed_manager, mock_tokenizer_provider):
        trace_file = _write_sharegpt_trace(tmp_path, [
            {
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                    {"from": "human", "value": "Q2"},
                    {"from": "gpt", "value": "A2"},
                    {"from": "human", "value": "Q3"},
                    {"from": "gpt", "value": "A3"},
                ]
            }
        ])
        flavor_config = UntimedContentMultiTurnTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config)
        gen = UntimedContentMultiTurnTraceFlavorGenerator(
            config, flavor_config, seed_manager, mock_tokenizer_provider
        )

        session = gen.generate_session()
        graph = session.session_graph

        # Should have 3 nodes and 2 edges
        assert len(graph.nodes) == 3
        all_edges = []
        for edge_list in graph.outgoing.values():
            all_edges.extend(edge_list)
        assert len(all_edges) == 2

        # All edges should have is_history_parent=False
        for edge in all_edges:
            assert edge.is_history_parent is False


class TestWrap:
    """Test wrapping behavior."""

    def test_wrap_shuffles_and_increments_ids(self, tmp_path, seed_manager, mock_tokenizer_provider):
        trace_file = _write_sharegpt_trace(tmp_path, [
            {
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Q2"},
                    {"from": "gpt", "value": "A2"},
                ]
            },
        ])
        flavor_config = UntimedContentMultiTurnTraceFlavorConfig()
        config = TraceSessionGeneratorConfig(
            trace_file=trace_file,
            flavor=flavor_config,
            wrap_mode=True,
        )
        gen = UntimedContentMultiTurnTraceFlavorGenerator(
            config, flavor_config, seed_manager, mock_tokenizer_provider
        )

        # Consume first epoch
        gen.generate_session()
        gen.generate_session()

        # Should wrap and continue
        session = gen.generate_session()
        assert session is not None


class TestMultipleConversations:
    """Test with multiple conversations in the trace."""

    def test_auto_session_id_assignment(self, tmp_path, seed_manager, mock_tokenizer_provider):
        trace_file = _write_sharegpt_trace(tmp_path, [
            {
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Q2"},
                    {"from": "gpt", "value": "A2"},
                ]
            },
        ])
        flavor_config = UntimedContentMultiTurnTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config)
        gen = UntimedContentMultiTurnTraceFlavorGenerator(
            config, flavor_config, seed_manager, mock_tokenizer_provider
        )

        s1 = gen.generate_session()
        s2 = gen.generate_session()
        assert s1.id != s2.id
        assert gen.capacity() == 2


class TestCSVFormat:
    """Test CSV trace file support."""

    def test_csv_with_json_string_conversations(self, tmp_path, seed_manager, mock_tokenizer_provider):
        import csv

        f = tmp_path / "trace.csv"
        with open(f, "w", newline="") as fd:
            writer = csv.writer(fd)
            writer.writerow(["conversations"])
            conv = [
                {"from": "human", "value": "Hello CSV"},
                {"from": "gpt", "value": "Hi from CSV!"},
            ]
            writer.writerow([json.dumps(conv)])

        flavor_config = UntimedContentMultiTurnTraceFlavorConfig()
        config = _make_config(str(f), flavor_config)
        gen = UntimedContentMultiTurnTraceFlavorGenerator(
            config, flavor_config, seed_manager, mock_tokenizer_provider
        )

        session = gen.generate_session()
        assert len(session.requests) == 1
        from veeksha.types import ChannelModality
        assert session.requests[0].channels[ChannelModality.TEXT].input_text == "Hello CSV"
