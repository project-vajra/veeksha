import json

import numpy as np
import pytest
import soundfile as sf

from scripts.align_audio_trace import (
    NeMoForcedAligner,
    attach_word_timings,
    build_alignment_plan,
)


@pytest.mark.unit
def test_attach_word_timings_maps_ctm_to_manifest_row(tmp_path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    sf.write(trace_dir / "clip.wav", np.zeros(16000, dtype=np.float32), 16000)

    manifest = trace_dir / "manifest.jsonl"
    rows = [
        {
            "audio_file": "clip.wav",
            "dataset": "toy",
            "expected_transcript": "hello world",
        },
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    plan = build_alignment_plan(manifest, tmp_path / "alignment")
    assert len(plan.items) == 1
    assert plan.items[0].row_index == 0

    plan.nemo_output_dir.mkdir()
    ctm_path = plan.nemo_output_dir / "clip.ctm"
    ctm_path.write_text(
        "utt 1 0.100 0.200 hello\n" "utt 1 1.200 0.300 world\n",
        encoding="utf-8",
    )
    (plan.nemo_output_dir / "nemo_manifest_with_output_file_paths.json").write_text(
        json.dumps({"word_ctm": str(ctm_path)}) + "\n",
        encoding="utf-8",
    )

    aligned_rows = attach_word_timings(plan)

    assert aligned_rows[0]["reference_word_timestamps"] == [
        {"word": "hello", "start_ms": 100.0, "end_ms": 300.0},
        {"word": "world", "start_ms": 1200.0, "end_ms": 1500.0},
    ]


@pytest.mark.unit
def test_nemo_forced_aligner_command_is_fixed_shape(tmp_path) -> None:
    align_script = tmp_path / "align.py"
    align_script.write_text("", encoding="utf-8")

    command = NeMoForcedAligner(align_script).command(
        tmp_path / "manifest.jsonl",
        tmp_path / "out",
    )

    assert str(align_script) in command
    assert "pretrained_name=stt_en_fastconformer_hybrid_large_pc" in command
    assert 'save_output_file_formats=["ctm"]' in command
