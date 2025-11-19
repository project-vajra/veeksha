import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytest

from veeksha.config.client import ClientConfig
from veeksha.config.generators.request_generator.trace_generator import (
    TraceRequestGeneratorConfig,
)
from veeksha.config.generators.session_generator import SessionGeneratorConfig
from veeksha.config.generators.interval_generator.static_generator import (
    StaticRequestIntervalGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.generators.request_generator.trace_generator import TraceRequestGenerator
from veeksha.logger import init_logger

logger = init_logger(__name__)

class _RoundTripTokenizer:
    """Minimal tokenizer with invertible encode/decode for ASCII text."""
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [ord(ch) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _write_trace_jsonl(
    tmp_path: Path,
    rows: List[dict],
    filename: str = "trace.jsonl"
) -> str:
    path = tmp_path / filename
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(path)


def _create_base_config(
    trace_file: str,
    session_generator_config: Optional[SessionGeneratorConfig] = None,
    use_trace_prefix_hash_ids: bool = False,
    corpus_lines: Optional[List[str]] = None,
) -> TraceRequestGeneratorConfig:
    return TraceRequestGeneratorConfig(
        trace_file=trace_file,
        input_length_column="input_length",
        output_length_column="output_length",
        block_size=512,
        timestamp_column="timestamp",
        timestamp_unit="ms",
        time_scale_factor=1.0,
        use_trace_prefix_hash_ids=use_trace_prefix_hash_ids,
        remap_hash_ids=False,
        session_generator_config=session_generator_config,
        max_tokens=4096,
        exhaustion_policy="error",
    )


def _build_generator(
    config: TraceRequestGeneratorConfig,
    corpus_lines: Optional[List[str]] = None,
) -> TraceRequestGenerator:
    tokenizer = _RoundTripTokenizer()
    client_cfg = ClientConfig()
    sm = SeedManager(123)
    
    if corpus_lines is None and not config.use_trace_prefix_hash_ids:
        corpus_lines = ["line1", "line2"]

    return TraceRequestGenerator(
        config=config,
        tokenizer=tokenizer,
        client_config=client_cfg,
        seed_manager=sm,
        corpus_lines=corpus_lines,
    )


@pytest.mark.unit
def test_legacy_trace_no_sessions(tmp_path):
    """Test a trace without session_id is treated as unique sessions of size 1."""
    rows = [
        {"input_length": 10, "output_length": 5, "timestamp": 1000},
        {"input_length": 20, "output_length": 5, "timestamp": 2000},
    ]
    trace_path = _write_trace_jsonl(tmp_path, rows)
    
    config = _create_base_config(trace_path)
    gen = _build_generator(config)
    
    # Request 1
    req1 = gen.get_request()
    assert req1.session_sequence_index == 0
    assert req1.session_id == 0  # global request ID
    assert req1.arrival_time is not None
    # Should use absolute timestamp from trace as arrival time (1000ms = 1.0s)
    assert req1.arrival_time == pytest.approx(1.0) 
    # Token count check: input(10)
    # The generator attempts to match input_length exactly, including instruction tokens.
    # If input_length is large enough to hold instructions, total length == input_length.
    # If input_length is too small, total length > input_length.
    # Here, 10 is likely enough or exactly enough.
    # Total length should be exactly max(input_length, instruction_length)
    # We assert it's reasonably close to input_length to catch runaway sizes.
    assert 10 <= req1.prompt[1] <= 10 + 50 # 50 is generous overhead buffer for instructions

    # Request 2
    req2 = gen.get_request()
    assert req2.session_sequence_index == 0
    assert req2.session_id == 1  # global request ID increments
    assert req2.arrival_time == pytest.approx(2.0)


@pytest.mark.unit
def test_explicit_session_trace_auto_detect(tmp_path):
    """Test a trace with explicit session_id is auto-detected and used."""
    rows = [
        # Session 100, Request 0
        {"session_id": 100, "timestamp": 1000, "input_length": 10, "output_length": 5},
        # Session 100, Request 1
        {"session_id": 100, "timestamp": 1500, "input_length": 10, "output_length": 5},
    ]
    trace_path = _write_trace_jsonl(tmp_path, rows)
    
    config = _create_base_config(trace_path)
    gen = _build_generator(config)
    
    # Request 1
    req1 = gen.get_request()
    assert req1.session_id == 100
    assert req1.session_sequence_index == 0
    # First in session uses anchor/absolute arrival
    assert req1.arrival_time == pytest.approx(1.0)
    
    # Request 2
    req2 = gen.get_request()
    assert req2.session_id == 100
    assert req2.session_sequence_index == 1
    # Subsequent request uses relative delay (think time)
    # 1500ms - 1000ms = 500ms = 0.5s
    assert req2.delay == pytest.approx(0.5)
    assert req2.arrival_time is None  # subsequent requests are relative


@pytest.mark.unit
def test_generated_sessions(tmp_path):
    """Test session generation configuration is respected."""
    # Rows with hash_ids for prefix matching
    rows = [
        {"input_length": 64, "output_length": 10, "timestamp": 1000, "hash_ids": [1, 2, 3]},
        {"input_length": 64, "output_length": 10, "timestamp": 2000, "hash_ids": [1, 2, 3, 4]},
    ]
    trace_path = _write_trace_jsonl(tmp_path, rows)
    
    session_cfg = SessionGeneratorConfig(
        minimum_prefix_match=0.5,
        min_session_size=2,
        max_session_size=2,
        max_request_interval=60.0,
        save_as_trace_file=False,
        session_interval_generator_config=StaticRequestIntervalGeneratorConfig(),
    )
    
    config = _create_base_config(
        trace_path, 
        session_generator_config=session_cfg,
        use_trace_prefix_hash_ids=True
    )
    gen = _build_generator(config)
    
    # We expect 1 session with 2 requests because prefixes match [1,2,3]
    req1 = gen.get_request()
    req2 = gen.get_request()
    
    assert req1.session_id == req2.session_id
    assert req1.session_sequence_index == 0
    assert req2.session_sequence_index == 1
    
    # Timestamps are re-generated/processed by SessionGenerator, 
    # but behavior should be consistent: first absolute, second relative.
    assert req1.arrival_time is not None
    assert req2.arrival_time is None
    assert req2.delay >= 0.0


@pytest.mark.unit
def test_exhaustion_wrap_with_sessions(tmp_path):
    """Test wrapping behavior correctly offsets session IDs."""
    rows = [
        {"session_id": 10, "timestamp": 1000, "input_length": 10, "output_length": 5}
    ]
    trace_path = _write_trace_jsonl(tmp_path, rows)
    
    config = TraceRequestGeneratorConfig(
        trace_file=trace_path,
        input_length_column="input_length",
        output_length_column="output_length",
        timestamp_column="timestamp",
        exhaustion_policy="wrap",
        max_tokens=4096,
    )
    gen = _build_generator(config)
    
    # Epoch 0
    req1 = gen.get_request()
    assert req1.session_id == 10
    
    # Epoch 1 (wrap)
    req2 = gen.get_request()
    # Should be offset by num_sessions_per_epoch (1 unique session in trace)
    # session_id = original (10) + offset (1 * 1) = 11
    assert req2.session_id == 11
    assert req2.session_sequence_index == 0


@pytest.mark.unit
def test_prompt_assembly_with_hash_ids(tmp_path):
    """Test prompt assembly using hash_ids (simulating prefix cache)."""
    # 2 requests sharing a common prefix hash [1, 2]
    rows = [
        {"input_length": 20, "output_length": 10, "timestamp": 1000, "hash_ids": [1, 2]},
        {"input_length": 20, "output_length": 10, "timestamp": 2000, "hash_ids": [1, 2, 3]},
    ]
    trace_path = _write_trace_jsonl(tmp_path, rows)

    config = _create_base_config(
        trace_path,
        use_trace_prefix_hash_ids=True
    )
    gen = _build_generator(config)

    req1 = gen.get_request()
    req2 = gen.get_request()

    # Validate prompt content
    # Prompts are assembled from cached blocks for hash IDs
    prompt1_txt, prompt1_len = req1.prompt
    prompt2_txt, prompt2_len = req2.prompt

    # Check that prompt2 starts with prompt1's prefix (minus instruction overhead if any)
    # Since hash_ids [1, 2] is a prefix of [1, 2, 3], the body text should be a prefix
    # Note: Instruction text might differ if output_length differs, but here it's same (10).
    assert prompt2_txt.startswith(prompt1_txt)
    assert prompt2_len > prompt1_len
    
    # Verify sampling params
    assert req1.sampling_params["max_completion_tokens"] == 10
    assert req2.sampling_params["max_completion_tokens"] == 10


@pytest.mark.unit
def test_sampling_params_override(tmp_path):
    """Test that client config sampling params are merged correctly."""
    rows = [
        {"input_length": 10, "output_length": 5, "timestamp": 1000},
    ]
    trace_path = _write_trace_jsonl(tmp_path, rows)
    
    config = _create_base_config(trace_path)
    
    # Inject custom client config
    tokenizer = _RoundTripTokenizer()
    client_cfg = ClientConfig(
        additional_sampling_params='{"temperature": 0.7, "top_p": 0.9}'
    )
    sm = SeedManager(123)
    gen = TraceRequestGenerator(
        config=config,
        tokenizer=tokenizer,
        client_config=client_cfg,
        seed_manager=sm,
        corpus_lines=["line1"]
    )

    req = gen.get_request()
    assert req.sampling_params["temperature"] == 0.7
    assert req.sampling_params["top_p"] == 0.9
    # max_completion_tokens should come from trace output_length
    assert req.sampling_params["max_completion_tokens"] == 5

