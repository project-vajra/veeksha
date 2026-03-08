"""Validate generated request traces for each trace flavor.

For each flavor, we create a realistic trace file, generate sessions,
and validate the output structure: correct number of requests, graph
topology, history population, and token counts.
"""

import json

import pytest
from unittest.mock import MagicMock

from veeksha.config.generator.session import (
    RequestLogTraceFlavorConfig,
    UntimedContentMultiTurnTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import parents
from veeksha.generator.session.trace.request_log import (
    RequestLogTraceFlavorGenerator,
)
from veeksha.generator.session.trace.conversation import (
    UntimedContentMultiTurnTraceFlavorGenerator,
)
from veeksha.types import ChannelModality


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _all_edges(session: Session):
    """Collect all edges from session graph."""
    edges = []
    for edge_list in session.session_graph.outgoing.values():
        edges.extend(edge_list)
    return edges


def _make_tokenizer_provider(*, vocab_size=1000):
    """Create a mock tokenizer provider with simple word-split tokenization."""
    provider = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = lambda text: text.split() if text else []
    tokenizer.decode = lambda ids: " ".join(str(i) for i in ids)
    tokenizer.count_tokens = lambda text: len(text.split()) if text else 0
    vocab = {f"tok_{i}": i for i in range(vocab_size)}
    tokenizer.get_vocab.return_value = vocab
    provider.for_modality.return_value = tokenizer
    return provider


def _make_config(trace_file, flavor_config, wrap_mode=False):
    return TraceSessionGeneratorConfig(
        trace_file=str(trace_file),
        flavor=flavor_config,
        wrap_mode=wrap_mode,
    )


def _validate_session_graph_topology(session: Session):
    """Validate basic session graph properties."""
    graph = session.session_graph
    nodes = graph.nodes
    edges = _all_edges(session)

    # All request keys match node IDs
    assert set(session.requests.keys()) == set(nodes.keys())

    # All edges reference valid nodes
    for edge in edges:
        assert edge.src in nodes, f"Edge src {edge.src} not in nodes"
        assert edge.dst in nodes, f"Edge dst {edge.dst} not in nodes"

    # Root nodes (no incoming edges) exist
    roots = [nid for nid in nodes if len(parents(graph, nid)) == 0]
    assert len(roots) >= 1, "No root nodes found"


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# request_log flavor
# ---------------------------------------------------------------------------

class TestRequestLogTraceGeneration:
    """Validate request_log flavor generates correct single-request sessions."""

    @pytest.fixture
    def trace_file(self, tmp_path):
        return _write_jsonl(tmp_path / "trace.jsonl", [
            {"input_length": 10, "output_length": 5},
            {"input_length": 20, "output_length": 8},
            {"input_length": 15, "output_length": 3},
        ])

    def test_generates_independent_sessions(self, trace_file):
        flavor_config = RequestLogTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config)
        gen = RequestLogTraceFlavorGenerator(
            config, flavor_config, SeedManager(seed=42), _make_tokenizer_provider()
        )

        sessions = [gen.generate_session() for _ in range(3)]

        for session in sessions:
            # Each session is a single request
            assert len(session.requests) == 1
            _validate_session_graph_topology(session)

            # No edges in single-request sessions
            edges = _all_edges(session)
            assert len(edges) == 0

            # Request has output spec
            req = session.requests[0]
            assert req.requested_output is not None
            assert req.requested_output.text.target_tokens > 0

            # Request has text content
            assert ChannelModality.TEXT in req.channels
            assert len(req.channels[ChannelModality.TEXT].input_text) > 0

    def test_output_tokens_match_trace(self, trace_file):
        flavor_config = RequestLogTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config)
        gen = RequestLogTraceFlavorGenerator(
            config, flavor_config, SeedManager(seed=42), _make_tokenizer_provider()
        )

        s1 = gen.generate_session()
        s2 = gen.generate_session()
        s3 = gen.generate_session()

        assert s1.requests[0].requested_output.text.target_tokens == 5
        assert s2.requests[0].requested_output.text.target_tokens == 8
        assert s3.requests[0].requested_output.text.target_tokens == 3

    def test_unique_session_ids(self, trace_file):
        flavor_config = RequestLogTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config)
        gen = RequestLogTraceFlavorGenerator(
            config, flavor_config, SeedManager(seed=42), _make_tokenizer_provider()
        )

        sessions = [gen.generate_session() for _ in range(3)]
        ids = [s.id for s in sessions]
        assert len(set(ids)) == 3

    def test_capacity_matches_trace_size(self, trace_file):
        flavor_config = RequestLogTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config, wrap_mode=False)
        gen = RequestLogTraceFlavorGenerator(
            config, flavor_config, SeedManager(seed=42), _make_tokenizer_provider()
        )
        assert gen.capacity() == 3

    def test_wrap_produces_more_sessions(self, trace_file):
        flavor_config = RequestLogTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config, wrap_mode=True)
        gen = RequestLogTraceFlavorGenerator(
            config, flavor_config, SeedManager(seed=42), _make_tokenizer_provider()
        )

        # Generate more sessions than trace size
        sessions = [gen.generate_session() for _ in range(6)]
        assert len(sessions) == 6


# ---------------------------------------------------------------------------
# untimed_content_multi_turn flavor
# ---------------------------------------------------------------------------

class TestUntimedContentMultiTurnTraceGeneration:
    """Validate untimed_content_multi_turn generates correct multi-turn sessions."""

    SHAREGPT_TRACE = [
        {
            "conversations": [
                {"from": "human", "value": "What is Python?"},
                {"from": "gpt", "value": "Python is a programming language created by Guido."},
                {"from": "human", "value": "Tell me about its history."},
                {"from": "gpt", "value": "Python was first released in 1991."},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "Hello world"},
                {"from": "gpt", "value": "Hi there! How can I help you?"},
            ]
        },
    ]

    LMSYS_TRACE = [
        {
            "conversation": [
                {"role": "user", "content": "Explain quantum computing"},
                {"role": "assistant", "content": "Quantum computing uses qubits."},
                {"role": "user", "content": "How is it different from classical?"},
                {"role": "assistant", "content": "Classical bits are 0 or 1."},
            ]
        },
    ]

    @pytest.fixture
    def sharegpt_trace(self, tmp_path):
        return _write_jsonl(tmp_path / "sharegpt.jsonl", self.SHAREGPT_TRACE)

    @pytest.fixture
    def lmsys_trace(self, tmp_path):
        return _write_jsonl(tmp_path / "lmsys.jsonl", self.LMSYS_TRACE)

    def _make_generator(self, trace_file, flavor_config=None, wrap_mode=False):
        if flavor_config is None:
            flavor_config = UntimedContentMultiTurnTraceFlavorConfig()
        config = _make_config(trace_file, flavor_config, wrap_mode=wrap_mode)
        return UntimedContentMultiTurnTraceFlavorGenerator(
            config, flavor_config, SeedManager(seed=42), _make_tokenizer_provider()
        )

    def test_sharegpt_session_structure(self, sharegpt_trace):
        gen = self._make_generator(sharegpt_trace)

        # Session 1: 2-turn conversation
        s1 = gen.generate_session()
        assert len(s1.requests) == 2
        _validate_session_graph_topology(s1)

        # Session 2: 1-turn conversation
        s2 = gen.generate_session()
        assert len(s2.requests) == 1
        _validate_session_graph_topology(s2)

    def test_history_pre_population(self, sharegpt_trace):
        """Verify request.history is correctly pre-populated from dataset."""
        gen = self._make_generator(sharegpt_trace)
        session = gen.generate_session()

        # Request 0: no history
        req0 = session.requests[0]
        assert req0.history == []
        assert req0.channels[ChannelModality.TEXT].input_text == "What is Python?"

        # Request 1: history contains first user+assistant exchange
        req1 = session.requests[1]
        assert len(req1.history) == 2
        assert req1.history[0]["role"] == "user"
        assert req1.history[0]["content"] == "What is Python?"
        assert req1.history[1]["role"] == "assistant"
        assert "programming language" in req1.history[1]["content"]
        assert req1.channels[ChannelModality.TEXT].input_text == "Tell me about its history."

    def test_is_history_parent_false_on_all_edges(self, sharegpt_trace):
        """All edges should have is_history_parent=False."""
        gen = self._make_generator(sharegpt_trace)
        session = gen.generate_session()

        edges = _all_edges(session)
        assert len(edges) == 1  # 2 requests → 1 edge
        for edge in edges:
            assert edge.is_history_parent is False

    def test_target_output_tokens_from_assistant_text(self, sharegpt_trace):
        """Output tokens should be computed by tokenizing assistant response."""
        gen = self._make_generator(sharegpt_trace)
        session = gen.generate_session()

        # Mock tokenizer splits on spaces
        # "Python is a programming language created by Guido." = 8 words
        req0 = session.requests[0]
        expected_tokens = len("Python is a programming language created by Guido.".split())
        assert req0.requested_output.text.target_tokens == expected_tokens

    def test_target_prompt_tokens_include_history(self, sharegpt_trace):
        """Prompt token count should include history + current user message."""
        gen = self._make_generator(sharegpt_trace)
        session = gen.generate_session()

        # Request 0: just user message tokens
        req0 = session.requests[0]
        assert req0.channels[ChannelModality.TEXT].target_prompt_tokens == len(
            "What is Python?".split()
        )

        # Request 1: history + user message tokens
        req1 = session.requests[1]
        # History: "What is Python?" + " " + "Python is a programming language created by Guido."
        # + " " + "Tell me about its history."
        history_text = "What is Python? Python is a programming language created by Guido."
        full_text = history_text + " " + "Tell me about its history."
        assert req1.channels[ChannelModality.TEXT].target_prompt_tokens == len(
            full_text.split()
        )

    def test_lmsys_schema(self, lmsys_trace):
        """LMSYS-Chat format with custom schema keys."""
        flavor_config = UntimedContentMultiTurnTraceFlavorConfig(
            conversation_column="conversation",
            role_key="role",
            content_key="content",
            user_role_value="user",
            assistant_role_value="assistant",
        )
        gen = self._make_generator(lmsys_trace, flavor_config)
        session = gen.generate_session()

        assert len(session.requests) == 2
        _validate_session_graph_topology(session)

        req0 = session.requests[0]
        assert req0.channels[ChannelModality.TEXT].input_text == "Explain quantum computing"
        assert req0.history == []

        req1 = session.requests[1]
        assert req1.history[0]["content"] == "Explain quantum computing"
        assert req1.history[1]["content"] == "Quantum computing uses qubits."

    def test_linear_graph_structure(self, sharegpt_trace):
        """Session graph should be linear: 0 → 1 → 2 → ..."""
        gen = self._make_generator(sharegpt_trace)
        session = gen.generate_session()
        graph = session.session_graph

        # Node 0 has no parents
        assert len(parents(graph, 0)) == 0

        # Node 1 has node 0 as parent
        parent_edges = parents(graph, 1)
        assert len(parent_edges) == 1
        assert parent_edges[0].src == 0

    def test_capacity_equals_conversation_count(self, sharegpt_trace):
        gen = self._make_generator(sharegpt_trace, wrap_mode=False)
        assert gen.capacity() == 2

    def test_wrap_continues_generating(self, sharegpt_trace):
        gen = self._make_generator(sharegpt_trace, wrap_mode=True)

        # Generate more sessions than trace contains
        sessions = [gen.generate_session() for _ in range(5)]
        assert len(sessions) == 5

    def test_edge_case_empty_conversations_skipped(self, tmp_path):
        """Empty conversation still produces a (minimal) session."""
        trace_file = _write_jsonl(tmp_path / "empty.jsonl", [
            {"conversations": []},
        ])
        gen = self._make_generator(trace_file)
        session = gen.generate_session()
        # Minimal fallback session
        assert len(session.requests) >= 1

    def test_edge_case_system_messages_skipped(self, tmp_path):
        """System messages don't produce requests."""
        trace_file = _write_jsonl(tmp_path / "system.jsonl", [
            {
                "conversations": [
                    {"from": "system", "value": "You are helpful."},
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "Hello!"},
                ]
            },
        ])
        gen = self._make_generator(trace_file)
        session = gen.generate_session()
        assert len(session.requests) == 1

    def test_edge_case_leading_assistant_skipped(self, tmp_path):
        """Leading assistant messages don't produce requests."""
        trace_file = _write_jsonl(tmp_path / "leading.jsonl", [
            {
                "conversations": [
                    {"from": "gpt", "value": "Welcome!"},
                    {"from": "human", "value": "Thanks"},
                    {"from": "gpt", "value": "You're welcome!"},
                ]
            },
        ])
        gen = self._make_generator(trace_file)
        session = gen.generate_session()
        assert len(session.requests) == 1
        assert session.requests[0].channels[ChannelModality.TEXT].input_text == "Thanks"

    def test_edge_case_trailing_user_skipped(self, tmp_path):
        """Trailing user message without response is skipped."""
        trace_file = _write_jsonl(tmp_path / "trailing.jsonl", [
            {
                "conversations": [
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "Hello!"},
                    {"from": "human", "value": "Unanswered"},
                ]
            },
        ])
        gen = self._make_generator(trace_file)
        session = gen.generate_session()
        assert len(session.requests) == 1

    def test_multi_turn_history_accumulates(self, tmp_path):
        """Verify history grows correctly across 4 turns."""
        trace_file = _write_jsonl(tmp_path / "multi.jsonl", [
            {
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                    {"from": "human", "value": "Q2"},
                    {"from": "gpt", "value": "A2"},
                    {"from": "human", "value": "Q3"},
                    {"from": "gpt", "value": "A3"},
                    {"from": "human", "value": "Q4"},
                    {"from": "gpt", "value": "A4"},
                ]
            },
        ])
        gen = self._make_generator(trace_file)
        session = gen.generate_session()

        assert len(session.requests) == 4

        # Turn 0: no history
        assert session.requests[0].history == []

        # Turn 1: 2 history messages (Q1, A1)
        assert len(session.requests[1].history) == 2

        # Turn 2: 4 history messages (Q1, A1, Q2, A2)
        assert len(session.requests[2].history) == 4

        # Turn 3: 6 history messages (Q1, A1, Q2, A2, Q3, A3)
        assert len(session.requests[3].history) == 6

        # Verify history content order
        h3 = session.requests[3].history
        assert h3[0] == {"role": "user", "content": "Q1"}
        assert h3[1] == {"role": "assistant", "content": "A1"}
        assert h3[4] == {"role": "user", "content": "Q3"}
        assert h3[5] == {"role": "assistant", "content": "A3"}

    def test_csv_format_with_json_strings(self, tmp_path):
        """CSV files with JSON-encoded conversation column."""
        import csv

        f = tmp_path / "trace.csv"
        with open(f, "w", newline="") as fd:
            writer = csv.writer(fd)
            writer.writerow(["conversations"])
            conv = [
                {"from": "human", "value": "CSV question"},
                {"from": "gpt", "value": "CSV answer"},
            ]
            writer.writerow([json.dumps(conv)])

        gen = self._make_generator(str(f))
        session = gen.generate_session()
        assert len(session.requests) == 1
        assert session.requests[0].channels[ChannelModality.TEXT].input_text == "CSV question"

    def test_request_ids_are_globally_unique(self, sharegpt_trace):
        """Request IDs should be unique across all sessions."""
        gen = self._make_generator(sharegpt_trace)
        s1 = gen.generate_session()
        s2 = gen.generate_session()

        all_ids = [r.id for r in s1.requests.values()] + [
            r.id for r in s2.requests.values()
        ]
        assert len(set(all_ids)) == len(all_ids)
