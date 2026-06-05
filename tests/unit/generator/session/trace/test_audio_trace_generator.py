import json
from unittest.mock import MagicMock

import pytest

from veeksha.config.generator.session import (
    AudioTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.generator.session.trace.audio import AudioTraceFlavorGenerator


@pytest.mark.unit
def test_audio_trace_metadata_preserves_reference_word_timestamp_list(tmp_path) -> None:
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"")
    reference_word_timestamps = [
        {"word": "hello", "start_ms": 0.0, "end_ms": 100.0},
        {"word": "world", "start_ms": 120.0, "end_ms": 300.0},
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "session_id": 0,
                "audio_file": audio_path.name,
                "expected_transcript": "hello world",
                "reference_word_timestamps": reference_word_timestamps,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    flavor_config = AudioTraceFlavorConfig()
    config = TraceSessionGeneratorConfig(
        trace_file=str(manifest_path),
        flavor=flavor_config,
        wrap_mode=False,
    )
    tokenizer_provider = MagicMock()
    tokenizer_provider.for_modality.return_value = MagicMock()

    generator = AudioTraceFlavorGenerator(
        config,
        flavor_config,
        SeedManager(seed=42),
        tokenizer_provider,
    )
    session = generator.generate_session()

    assert session.requests[0].metadata["reference_word_timestamps"] == (
        reference_word_timestamps
    )
