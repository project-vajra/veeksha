from __future__ import annotations

import pytest

from veeksha.evaluator.performance.asr import (
    ASRMetricAccumulator,
    WERStats,
    compute_wer_stats,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "hypothesis",
    [
        "alpha delta charlie",  # substitution
        "alpha charlie",  # deletion
        "alpha bravo extra charlie",  # insertion
    ],
)
def test_asr_wer_counts_each_edit_class_against_a_manual_oracle(
    hypothesis: str,
) -> None:
    stats = compute_wer_stats("alpha bravo charlie", hypothesis)

    assert stats.errors == 1
    assert stats.reference_words == 3
    assert stats.wer == pytest.approx(100 / 3)


@pytest.mark.unit
def test_asr_wer_applies_the_leaderboard_normalizer_before_alignment() -> None:
    stats = compute_wer_stats(
        "Mr. Smith won't pay twenty-one dollars.",
        "mister smith will not pay 21 dollars",
    )

    assert stats == WERStats(errors=0, reference_words=6, wer=0.0)


@pytest.mark.unit
def test_asr_aggregates_distinguish_sample_corpus_and_duration_weighted_wer() -> None:
    accumulator = ASRMetricAccumulator()
    accumulator.add_clip_sample(
        dataset="short",
        duration_s=1.0,
        final_stats=WERStats(errors=1, reference_words=2, wer=50.0),
        partial_stats=None,
    )
    accumulator.add_clip_sample(
        dataset="long",
        duration_s=3.0,
        final_stats=WERStats(errors=1, reference_words=8, wer=12.5),
        partial_stats=None,
    )

    summary = accumulator.get_summary()

    assert summary["asr_final_sample_mean_wer"] == pytest.approx(31.25)
    assert summary["asr_final_corpus_wer"] == pytest.approx(20.0)
    assert summary["asr_final_duration_weighted_wer"] == pytest.approx(21.875)
    assert summary["asr_dataset_short_final_corpus_wer"] == pytest.approx(50.0)
    assert summary["asr_dataset_long_final_corpus_wer"] == pytest.approx(12.5)


@pytest.mark.unit
def test_asr_empty_reference_has_explicit_zero_or_full_error_semantics() -> None:
    assert compute_wer_stats("", "") == WERStats(errors=0, reference_words=0, wer=0.0)
    assert compute_wer_stats("", "unexpected") == WERStats(
        errors=1, reference_words=0, wer=100.0
    )

    accumulator = ASRMetricAccumulator()
    accumulator.add_clip_sample(
        dataset="empty",
        duration_s=1.0,
        final_stats=compute_wer_stats("", "unexpected"),
        partial_stats=None,
    )
    assert accumulator.get_summary()["asr_final_corpus_wer"] == 100.0
