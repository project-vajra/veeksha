"""Unit tests for position-resolved long-form TTS scoring."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from veeksha.cli.base import VeekshaCommand
from veeksha.config.score_tts_longform import (
    LongformSimConfig,
    ScoreTtsLongformConfig,
)
from veeksha.verification import longform as longform_verification
from veeksha.verification.audio import normalize_text
from veeksha.verification.longform import (
    AudioSource,
    align_chunks_to_reference,
    attribute_alignment,
    bucket_weighted,
    build_sim_embedder,
    compression_ratio,
    compute_energy_bins,
    count_duplicated_ngrams,
    make_chunk_spans,
    run_score_tts_longform,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeOp:
    """Stand-in for a jiwer AlignmentChunk."""

    type: str
    ref_start_idx: int
    ref_end_idx: int
    hyp_start_idx: int
    hyp_end_idx: int


class ScriptedTranscriber:
    """Returns pre-scripted transcripts, one per chunk in order."""

    def __init__(self, transcripts: list[str]):
        self._transcripts = list(transcripts)
        self._calls = 0

    def transcribe(self, wav_16k: np.ndarray) -> str:
        assert self._calls < len(self._transcripts), "More chunks than scripted"
        transcript = self._transcripts[self._calls]
        self._calls += 1
        return transcript


class PositionUtmosScorer:
    """Scores 4.0 before the pivot chunk index, 3.0 from it onward."""

    def __init__(self, pivot_call: int):
        self._pivot_call = pivot_call
        self._calls = 0

    def score(self, wav_16k: np.ndarray) -> Optional[float]:
        score = 4.0 if self._calls < self._pivot_call else 3.0
        self._calls += 1
        return score


class DriftingSimEmbedder:
    """Anchor + early windows embed to e1; from the pivot call onward, e2."""

    def __init__(self, pivot_call: int):
        self._pivot_call = pivot_call
        self._calls = 0

    def embed(self, wav_16k: np.ndarray) -> np.ndarray:
        embedding = (
            np.array([1.0, 0.0], dtype=np.float32)
            if self._calls < self._pivot_call
            else np.array([0.0, 1.0], dtype=np.float32)
        )
        self._calls += 1
        return embedding


def _write_pcm(path: Path, sample_rate: int, seconds: float) -> None:
    """Sine bursts alternating with silence: 5 s on, 5 s off."""
    num_samples = int(seconds * sample_rate)
    t = np.arange(num_samples) / sample_rate
    wave = 0.5 * np.sin(2 * math.pi * 220.0 * t)
    gate = (t % 10.0) < 5.0
    pcm = (wave * gate * 32767.0).astype(np.int16)
    pcm.tofile(path)


_REF_WORDS = [f"w{i:02d}" for i in range(30)]


def _make_config(tmp_path: Path, **overrides) -> ScoreTtsLongformConfig:
    audio_path = tmp_path / "audio.pcm"
    if not audio_path.exists():
        _write_pcm(audio_path, 8000, 70.0)
    reference_path = tmp_path / "reference.txt"
    if not reference_path.exists():
        reference_path.write_text(" ".join(_REF_WORDS), encoding="utf-8")
    defaults = dict(
        audio=str(audio_path),
        sample_rate=8000,
        reference_text=str(reference_path),
        output_dir=str(tmp_path / "scores"),
    )
    defaults.update(overrides)
    return ScoreTtsLongformConfig(**defaults)


# ---------------------------------------------------------------------------
# Seed-exact normalizer
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_strips_punctuation_except_apostrophe(self):
        assert normalize_text("Hello, World! It's fine.") == "hello world it's fine"

    def test_hyphen_removal_joins_words(self):
        assert normalize_text("state-of-the-art") == "stateoftheart"

    def test_double_space_collapsed(self):
        assert normalize_text("a  b") == "a b"

    def test_triple_space_single_pass_seed_quirk(self):
        # Seed's normalizer runs one replace pass: "a   b" -> "a  b".
        assert normalize_text("a   b") == "a  b"

    def test_digits_not_normalized(self):
        assert normalize_text("I have 3 dogs") == "i have 3 dogs"

    def test_lowercases(self):
        assert normalize_text("NINETY FIVE") == "ninety five"

    def test_punctuation_leaves_double_space(self):
        # Punctuation stripping can create "  ", which then collapses.
        assert normalize_text("stop. and go") == "stop and go"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class TestMakeChunkSpans:
    def test_exact_division(self):
        spans = make_chunk_spans(24000 * 56, 24000, 28.0)
        assert len(spans) == 2
        assert spans[0].start_sample == 0
        assert spans[0].end_sample == 24000 * 28
        assert spans[1].end_sample == 24000 * 56
        assert spans[1].start_s == pytest.approx(28.0)

    def test_partial_tail_kept_by_default(self):
        spans = make_chunk_spans(24000 * 30, 24000, 28.0)
        assert len(spans) == 2
        assert spans[1].duration_s == pytest.approx(2.0)

    def test_partial_tail_dropped_below_minimum(self):
        spans = make_chunk_spans(int(24000 * 10.5), 24000, 10.0, min_tail_seconds=1.0)
        assert len(spans) == 1

    def test_partial_tail_kept_above_minimum(self):
        spans = make_chunk_spans(int(24000 * 11.5), 24000, 10.0, min_tail_seconds=1.0)
        assert len(spans) == 2
        assert spans[1].duration_s == pytest.approx(1.5)

    def test_single_short_chunk(self):
        spans = make_chunk_spans(24000, 24000, 28.0)
        assert len(spans) == 1
        assert spans[0].duration_s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Repetition detectors
# ---------------------------------------------------------------------------


class TestRepetitionDetectors:
    def test_no_duplicates(self):
        words = [f"u{i}" for i in range(20)]
        assert count_duplicated_ngrams(words, 5) == 0

    def test_planted_duplicate_counted(self):
        words = ["a", "b", "c", "d", "e", "x", "a", "b", "c", "d", "e"]
        assert count_duplicated_ngrams(words, 5) == 1

    def test_triple_occurrence_counts_two(self):
        phrase = ["a", "b", "c", "d", "e"]
        words = phrase + ["x"] + phrase + ["y"] + phrase
        assert count_duplicated_ngrams(words, 5) == 2

    def test_short_transcript_zero(self):
        assert count_duplicated_ngrams(["a", "b"], 5) == 0

    def test_compression_ratio_flags_loops(self):
        looped = "the same phrase again " * 50
        assert compression_ratio(looped) > 2.4

    def test_compression_ratio_normal_text(self):
        normal = "they climbed a tall crane without securing themselves"
        assert compression_ratio(normal) < 2.4

    def test_compression_ratio_empty(self):
        assert compression_ratio("") == 0.0


# ---------------------------------------------------------------------------
# Global alignment attribution
# ---------------------------------------------------------------------------


class TestAlignmentAttribution:
    def test_perfect_transcript_zero_wer(self):
        chunks = [_REF_WORDS[:10], _REF_WORDS[10:20], _REF_WORDS[20:]]
        global_stats, counts = align_chunks_to_reference(
            _REF_WORDS, chunks, [28.0, 28.0, 14.0]
        )
        assert global_stats.wer == 0.0
        assert all(c.wer == 0.0 for c in counts)
        assert [c.ref_words for c in counts] == [10.0, 10.0, 10.0]

    def test_planted_errors_attributed_to_their_chunks(self):
        # chunk0: w00..w09 plus a duplicated w05..w09 tail (5 insertions).
        chunk0 = _REF_WORDS[:10] + _REF_WORDS[5:10]
        # chunk1: w10..w19 minus w12,w13 (2 deletions), w15 -> x15 (1 sub).
        chunk1 = [w for w in _REF_WORDS[10:20] if w not in ("w12", "w13")]
        chunk1[chunk1.index("w15")] = "x15"
        # chunk2: perfect.
        chunk2 = _REF_WORDS[20:]

        global_stats, counts = align_chunks_to_reference(
            _REF_WORDS, [chunk0, chunk1, chunk2], [28.0, 28.0, 14.0]
        )

        assert global_stats.insertions == 5
        assert global_stats.deletions == 2
        assert global_stats.substitutions == 1
        assert global_stats.hits == 27
        assert global_stats.wer == pytest.approx(8 / 30)

        assert counts[0].insertions == pytest.approx(5.0)
        assert counts[0].deletions == pytest.approx(0.0)
        assert counts[1].deletions == pytest.approx(2.0)
        assert counts[1].substitutions == pytest.approx(1.0)
        assert counts[1].wer == pytest.approx(3 / 10)
        assert counts[2].errors == pytest.approx(0.0)
        assert counts[2].wer == 0.0

    def test_all_silent_chunks_split_deletions_by_duration(self):
        chunks: list[list[str]] = [[], [], []]
        global_stats, counts = align_chunks_to_reference(
            _REF_WORDS, chunks, [28.0, 28.0, 14.0]
        )
        assert global_stats.deletions == 30
        assert counts[0].deletions == pytest.approx(30 * 28 / 70)
        assert counts[1].deletions == pytest.approx(30 * 28 / 70)
        assert counts[2].deletions == pytest.approx(30 * 14 / 70)

    def test_empty_middle_chunk_receives_deletion_share(self):
        ops = [
            FakeOp("equal", 0, 5, 0, 5),
            FakeOp("delete", 5, 9, 5, 5),
            FakeOp("equal", 9, 13, 5, 9),
        ]
        counts = attribute_alignment(ops, [0, 5, 5, 9], [28.0, 28.0, 14.0])
        # Deletion at position 5 splits between empty chunk 1 (28 s) and
        # containing chunk 2 (14 s), duration-proportionally.
        assert counts[0].deletions == pytest.approx(0.0)
        assert counts[1].deletions == pytest.approx(4 * 28 / 42)
        assert counts[2].deletions == pytest.approx(4 * 14 / 42)

    def test_op_spanning_chunk_boundary_splits(self):
        ops = [FakeOp("equal", 0, 10, 0, 10)]
        counts = attribute_alignment(ops, [0, 4, 10], [28.0, 28.0])
        assert counts[0].hits == pytest.approx(4.0)
        assert counts[1].hits == pytest.approx(6.0)

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError, match="Reference text has no words"):
            align_chunks_to_reference([], [["a"]], [28.0])


# ---------------------------------------------------------------------------
# Bucket assembly
# ---------------------------------------------------------------------------


class TestBucketWeighted:
    def test_chunk_straddling_bucket_boundary(self):
        # 28 s chunks vs 60 s buckets: chunk [56, 84) puts 4/28 of its mass
        # in bucket 0 and 24/28 in bucket 1.
        spans = [(0.0, 28.0), (28.0, 56.0), (56.0, 84.0)]
        values = [{"errors": 28.0}, {"errors": 28.0}, {"errors": 28.0}]
        buckets = bucket_weighted(spans, values, 60.0, 2)
        assert buckets[0]["errors"] == pytest.approx(28.0 + 28.0 + 28.0 * 4 / 28)
        assert buckets[1]["errors"] == pytest.approx(28.0 * 24 / 28)

    def test_conservation(self):
        spans = [(i * 28.0, (i + 1) * 28.0) for i in range(5)]
        values = [{"x": float(i)} for i in range(5)]
        buckets = bucket_weighted(spans, values, 60.0, 3)
        assert sum(b.get("x", 0.0) for b in buckets) == pytest.approx(
            sum(float(i) for i in range(5))
        )

    def test_span_exactly_on_boundary(self):
        buckets = bucket_weighted([(60.0, 90.0)], [{"x": 1.0}], 60.0, 2)
        assert buckets[0] == {}
        assert buckets[1]["x"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Energy track
# ---------------------------------------------------------------------------


class TestEnergyBins:
    def test_sine_then_silence(self, tmp_path: Path):
        sample_rate = 8000
        t = np.arange(sample_rate) / sample_rate
        sine = 0.5 * np.sin(2 * math.pi * 220.0 * t)
        silence = np.zeros(sample_rate)
        pcm = (np.concatenate([sine, silence]) * 32767.0).astype(np.int16)
        path = tmp_path / "tone.pcm"
        pcm.tofile(path)

        audio = AudioSource(path, sample_rate)
        bins = compute_energy_bins(audio, 1.0, -40.0)

        assert len(bins) == 2
        # RMS of a 0.5-amplitude sine is 0.5/sqrt(2) ~= -9.03 dBFS.
        assert bins[0].rms_dbfs == pytest.approx(-9.03, abs=0.1)
        assert bins[0].silence_frac == pytest.approx(0.0)
        assert bins[1].rms_dbfs == -100.0
        assert bins[1].silence_frac == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AudioSource
# ---------------------------------------------------------------------------


class TestAudioSource:
    def test_pcm_roundtrip(self, tmp_path: Path):
        pcm = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        path = tmp_path / "raw.pcm"
        pcm.tofile(path)
        audio = AudioSource(path, 24000)
        assert audio.num_samples == 4
        chunk = audio.slice_f32(0, 4)
        assert chunk.dtype == np.float32
        assert chunk[1] == pytest.approx(0.5, abs=1e-4)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            AudioSource(tmp_path / "nope.pcm", 24000)

    def test_empty_file_raises(self, tmp_path: Path):
        path = tmp_path / "empty.pcm"
        path.touch()
        with pytest.raises(ValueError, match="empty"):
            AudioSource(path, 24000)


# ---------------------------------------------------------------------------
# Full pipeline with injected fakes
# ---------------------------------------------------------------------------


class TestPipelineWithFakes:
    def _run(self, tmp_path: Path):
        config = _make_config(tmp_path)
        # 70 s at 8 kHz -> WER chunks [0,28), [28,56), [56,70).
        chunk0 = " ".join(_REF_WORDS[:10] + _REF_WORDS[5:10])
        chunk1 = " ".join(
            "x15" if w == "w15" else w
            for w in _REF_WORDS[10:20]
            if w not in ("w12", "w13")
        )
        chunk2 = " ".join(_REF_WORDS[20:])
        transcriber = ScriptedTranscriber([chunk0, chunk1, chunk2])
        # UTMOS chunks are 10 s -> 7 chunks; pivot at chunk 6 (start 60 s).
        utmos = PositionUtmosScorer(pivot_call=6)
        # SIM: anchor + 23 windows of 3 s; drift to orthogonal at call 21
        # (window starts 60 s).
        sim = DriftingSimEmbedder(pivot_call=21)
        return run_score_tts_longform(
            config, transcriber=transcriber, utmos_scorer=utmos, sim_embedder=sim
        )

    def test_outputs_written(self, tmp_path: Path):
        result = self._run(tmp_path)
        assert result.summary_path is not None and result.summary_path.exists()
        assert result.curves_path is not None and result.curves_path.exists()
        assert result.report_path is not None and result.report_path.exists()
        saved = json.loads(result.summary_path.read_text())
        assert saved["wer"]["global"]["wer"] == pytest.approx(8 / 30)

    def test_global_wer_and_repetition(self, tmp_path: Path):
        summary = self._run(tmp_path).summary
        global_wer = summary["wer"]["global"]
        assert global_wer["insertions"] == 5
        assert global_wer["deletions"] == 2
        assert global_wer["substitutions"] == 1
        repetition = summary["repetition"]
        assert repetition["total_dup_ngrams"] == 1
        assert repetition["flagged_chunks"] == 0

    def test_bucket_rows(self, tmp_path: Path):
        result = self._run(tmp_path)
        rows = result.bucket_rows
        assert len(rows) == 2  # 70 s -> buckets [0,60), [60,70)

        # Bucket 0: chunk0 fully (5 ins over 10 ref), chunk1 fully (3 errors
        # over 10 ref), 4/14 of chunk2 (perfect).
        expected_ref_bucket0 = 10.0 + 10.0 + 10.0 * (4 / 14)
        assert rows[0].wer == pytest.approx(8.0 / expected_ref_bucket0)
        assert rows[0].insertions == pytest.approx(5.0)
        assert rows[0].deletions == pytest.approx(2.0)
        assert rows[0].dup_ngrams == pytest.approx(1.0)
        assert rows[1].wer == pytest.approx(0.0)

        # UTMOS: bucket 0 has six 4.0 chunks; bucket 1 one 3.0 chunk.
        assert rows[0].utmos_mean == pytest.approx(4.0)
        assert rows[0].utmos_min == pytest.approx(4.0)
        assert rows[1].utmos_mean == pytest.approx(3.0)

        # SIM: windows starting at 60 s drift to cosine 0.
        assert rows[0].sim_mean == pytest.approx(1.0)
        assert rows[1].sim_mean == pytest.approx(0.0)
        assert rows[1].sim_min == pytest.approx(0.0)

        # Energy present everywhere.
        assert rows[0].rms_dbfs is not None
        assert rows[0].silence_frac == pytest.approx(0.5, abs=0.05)

    def test_curves_csv_shape(self, tmp_path: Path):
        result = self._run(tmp_path)
        with result.curves_path.open() as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = list(reader)
        assert header == [
            "bucket",
            "wer",
            "utmos_mean",
            "utmos_min",
            "ins",
            "del",
            "dup5grams",
            "rms",
            "silence_frac",
        ]
        assert len(rows) == 2
        assert rows[0][0] == "0"
        assert float(rows[0][1]) == pytest.approx(
            8.0 / (20.0 + 10.0 * 4 / 14), abs=1e-4
        )

    def test_report_mentions_tracks(self, tmp_path: Path):
        result = self._run(tmp_path)
        report = result.report_path.read_text()
        assert "WER:" in report
        assert "UTMOS: mean" in report
        assert "SIM: mean" in report
        assert "PER-BUCKET CURVES" in report


# ---------------------------------------------------------------------------
# Graceful degradation (no models available)
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_all_models_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            longform_verification,
            "build_transcriber",
            lambda _config: (None, "WER skipped: unavailable for test"),
        )
        monkeypatch.setattr(
            longform_verification,
            "build_utmos_scorer",
            lambda _config: (None, "UTMOS skipped: unavailable for test"),
        )
        monkeypatch.setattr(
            longform_verification,
            "build_sim_embedder",
            lambda _config: (None, "SIM skipped: checkpoint not available"),
        )
        config = _make_config(tmp_path)
        result = run_score_tts_longform(config)

        summary = result.summary
        assert summary["wer"] is None
        assert summary["utmos"] is None
        assert summary["sim"] is None
        notes = " | ".join(summary["notes"])
        assert "WER skipped" in notes
        assert "UTMOS skipped" in notes
        assert "SIM skipped: checkpoint not available" in notes

        assert summary["energy"]["global_rms_dbfs"] > -100.0
        assert 0.0 < summary["energy"]["global_silence_frac"] < 1.0

        with result.curves_path.open() as handle:
            reader = csv.reader(handle)
            next(reader)
            first = next(reader)
        assert first[1] == ""  # wer empty
        assert first[2] == ""  # utmos empty
        assert first[7] != ""  # rms populated

        report = result.report_path.read_text()
        assert "WER: skipped" in report
        assert "UTMOS: skipped" in report
        assert "SIM: skipped" in report

    def test_sim_missing_checkpoint_file_note(self, tmp_path: Path):
        embedder, note = build_sim_embedder(
            LongformSimConfig(checkpoint_path=str(tmp_path / "missing.pt"))
        )
        assert embedder is None
        assert "SIM skipped: checkpoint not available at" in note

    def test_sim_unset_checkpoint_note_points_to_source(self):
        embedder, note = build_sim_embedder(LongformSimConfig())
        assert embedder is None
        assert "checkpoint not available" in note
        assert "sim.checkpoint_path" in note


# ---------------------------------------------------------------------------
# Config validation and CLI registration
# ---------------------------------------------------------------------------


class TestConfigAndCli:
    def test_missing_audio_raises(self):
        with pytest.raises(ValueError, match="requires --audio"):
            ScoreTtsLongformConfig(reference_text="ref.txt")

    def test_missing_reference_raises(self):
        with pytest.raises(ValueError, match="requires --reference_text"):
            ScoreTtsLongformConfig(audio="a.pcm")

    def test_asr_chunk_over_30s_rejected(self, tmp_path: Path):
        from veeksha.config.score_tts_longform import LongformAsrConfig

        with pytest.raises(ValueError, match="30"):
            LongformAsrConfig(chunk_seconds=31.0)

    def test_subcommand_registered(self):
        assert (
            VeekshaCommand.resolve_subcommand("score-tts-longform")
            is ScoreTtsLongformConfig
        )

    def test_cli_help_exits_cleanly(self):
        completed = subprocess.run(
            [sys.executable, "-m", "veeksha", "score-tts-longform", "--help"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0
        assert "score a long-form tts waveform" in completed.stdout.lower()
        assert "--sim.checkpoint_path" in completed.stdout
        assert "UniSpeech" in completed.stdout
