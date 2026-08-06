import json

import numpy as np
import pytest
import soundfile as sf

from scripts.prepare_hf_audio_trace import (
    DEFAULT_REVISION,
    prepare_config,
    validate_revision,
)


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0])
        self.cast = None

    def cast_column(self, column, feature):
        self.cast = (column, feature)
        return self

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _row(index: int) -> dict:
    # Channel-first stereo at 8 kHz exercises both mono conversion and
    # resampling before the PCM16 file is written.
    audio = np.vstack(
        (
            np.full(80, 0.25 + index * 0.01, dtype=np.float32),
            np.full(80, -0.25, dtype=np.float32),
        )
    )
    return {
        "audio": {"array": audio, "sampling_rate": 8_000},
        "transcript": f"বাংলা নমুনা {index}",
        "language": "Bengali",
        "duration": 0.01,
        "dataset": "kathbath",
        "utt_id": f"utt-{index}",
    }


@pytest.mark.unit
def test_prepare_config_writes_portable_pcm16_trace_with_provenance(tmp_path) -> None:
    dataset = FakeDataset([_row(0), _row(1)])

    def loader(repo_id, config, *, split, revision):
        assert (repo_id, config, split, revision) == (
            "owner/repo",
            "kathbath",
            "test",
            DEFAULT_REVISION,
        )
        return dataset

    metadata = prepare_config(
        repo_id="owner/repo",
        revision=DEFAULT_REVISION,
        config="kathbath",
        split="test",
        output_dir=tmp_path,
        dataset_loader=loader,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "kathbath" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["expected_transcript"] for row in rows] == [
        "বাংলা নমুনা 0",
        "বাংলা নমুনা 1",
    ]
    assert rows[0] == {
        "audio_file": "audio/clip_000000.wav",
        "dataset": "kathbath",
        "duration": 0.01,
        "duration_s": 0.01,
        "expected_transcript": "বাংলা নমুনা 0",
        "language": "bn",
        "language_name": "Bengali",
        "session_id": 0,
        "source_id": "kathbath:utt-0",
        "source_revision": DEFAULT_REVISION,
        "source_row_index": 0,
    }

    wav_info = sf.info(tmp_path / "kathbath" / rows[0]["audio_file"])
    assert wav_info.samplerate == 16_000
    assert wav_info.channels == 1
    assert wav_info.subtype == "PCM_16"
    assert wav_info.frames == 160

    assert metadata["canonical"] is True
    assert metadata["selection"] == {
        "order": "source_order",
        "total_rows": 2,
        "materialized_rows": 2,
        "max_samples": None,
    }
    assert len(metadata["manifest"]["sha256"]) == 64


@pytest.mark.unit
def test_max_samples_is_a_deterministic_noncanonical_prefix(tmp_path) -> None:
    dataset = FakeDataset([_row(0), _row(1), _row(2)])
    metadata = prepare_config(
        repo_id="owner/repo",
        revision=DEFAULT_REVISION,
        config="kathbath",
        split="test",
        output_dir=tmp_path,
        max_samples=2,
        dataset_loader=lambda *args, **kwargs: dataset,
    )

    manifest_lines = (
        (tmp_path / "kathbath" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert [json.loads(line)["source_row_index"] for line in manifest_lines] == [0, 1]
    assert metadata["canonical"] is False
    assert metadata["selection"]["materialized_rows"] == 2
    assert "smoke testing" in metadata["noncanonical_reason"]


@pytest.mark.unit
def test_revision_must_be_an_exact_commit() -> None:
    with pytest.raises(ValueError, match="exact 40-character"):
        validate_revision("main")
