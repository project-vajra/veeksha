"""ASR-specific scoring helpers for realtime speech-to-text benchmarks."""

from __future__ import annotations

import threading
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Optional

import jiwer

from veeksha.evaluator.performance.asr_interactivity import compute_interactivity_stats
from veeksha.evaluator.performance.asr_normalizer import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


def normalize_unicode_text(text: str) -> str:
    """Minimal multilingual normalization that preserves Indic script marks.

    NFC canonicalizes equivalent code-point sequences. Letters, numbers, and
    combining marks are retained; punctuation, symbols, controls, and separator
    characters become word boundaries. Unlike the English leaderboard
    normalizer, this does not transliterate or delete Indic vowel signs.
    """

    normalized = unicodedata.normalize("NFC", text).casefold()
    characters: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if category[0] in {"L", "N", "M"}:
            characters.append(character)
        else:
            characters.append(" ")
    return " ".join("".join(characters).split())


def _normalize(text: str, normalizer: str) -> str:
    if normalizer == "english":
        return _normalizer(text)
    if normalizer == "unicode":
        return normalize_unicode_text(text)
    raise ValueError(f"Unsupported ASR text normalizer: {normalizer!r}")


@dataclass(frozen=True)
class WERStats:
    """Edit counts and WER percentage for one normalized transcript comparison."""

    errors: int
    reference_words: int
    wer: float


def compute_wer_stats(
    reference: str, hypothesis: str, *, normalizer: str = "english"
) -> WERStats:
    """WER counts and percentage using the leaderboard normalizer + jiwer."""
    ref = _normalize(reference, normalizer)
    hyp = _normalize(hypothesis, normalizer)
    output = jiwer.process_words(ref, hyp)
    errors = output.substitutions + output.deletions + output.insertions
    reference_words = output.hits + output.substitutions + output.deletions
    if reference_words == 0:
        wer = 0.0 if errors == 0 else 100.0
    else:
        wer = (errors / reference_words) * 100
    return WERStats(errors=errors, reference_words=reference_words, wer=wer)


@dataclass(frozen=True)
class CERStats:
    """Edit counts and CER percentage after the configured normalization."""

    errors: int
    reference_characters: int
    cer: float


def compute_cer_stats(
    reference: str, hypothesis: str, *, normalizer: str = "unicode"
) -> CERStats:
    """Character error rate excluding whitespace word-boundary characters."""

    ref = _normalize(reference, normalizer).replace(" ", "")
    hyp = _normalize(hypothesis, normalizer).replace(" ", "")
    output = jiwer.process_characters(ref, hyp)
    errors = output.substitutions + output.deletions + output.insertions
    reference_characters = output.hits + output.substitutions + output.deletions
    if reference_characters == 0:
        cer = 0.0 if errors == 0 else 100.0
    else:
        cer = (errors / reference_characters) * 100
    return CERStats(
        errors=errors,
        reference_characters=reference_characters,
        cer=cer,
    )


@dataclass
class WERAggregate:
    """Accumulates multiple WER aggregation modes for comparable ASR samples."""

    sample_count: int = 0
    wer_sum: float = 0.0
    duration_weighted_wer_sum: float = 0.0
    duration_s_sum: float = 0.0
    errors: int = 0
    reference_words: int = 0

    def add(self, stats: WERStats, duration_s: float) -> None:
        self.sample_count += 1
        self.wer_sum += stats.wer
        self.errors += stats.errors
        self.reference_words += stats.reference_words
        if duration_s > 0:
            self.duration_weighted_wer_sum += stats.wer * duration_s
            self.duration_s_sum += duration_s

    def summary(self, prefix: str) -> Dict[str, Optional[float]]:
        sample_mean = (
            self.wer_sum / self.sample_count if self.sample_count > 0 else None
        )
        corpus = (
            (self.errors / self.reference_words) * 100
            if self.reference_words > 0
            else (
                100.0 if self.errors > 0 else (0.0 if self.sample_count > 0 else None)
            )
        )
        duration_weighted = (
            self.duration_weighted_wer_sum / self.duration_s_sum
            if self.duration_s_sum > 0
            else None
        )
        return {
            f"{prefix}_sample_count": float(self.sample_count),
            # Preserve sufficient statistics so a named benchmark can combine
            # disjoint dataset runs exactly. Averaging per-dataset WERs would
            # overweight small datasets and is not corpus WER.
            f"{prefix}_errors": float(self.errors),
            f"{prefix}_reference_words": float(self.reference_words),
            f"{prefix}_sample_mean_wer": sample_mean,
            f"{prefix}_corpus_wer": corpus,
            f"{prefix}_duration_weighted_wer": duration_weighted,
        }


@dataclass
class CERAggregate:
    """Accumulates exact CER sufficient statistics across samples."""

    sample_count: int = 0
    cer_sum: float = 0.0
    duration_weighted_cer_sum: float = 0.0
    duration_s_sum: float = 0.0
    errors: int = 0
    reference_characters: int = 0

    def add(self, stats: CERStats, duration_s: float) -> None:
        self.sample_count += 1
        self.cer_sum += stats.cer
        self.errors += stats.errors
        self.reference_characters += stats.reference_characters
        if duration_s > 0:
            self.duration_weighted_cer_sum += stats.cer * duration_s
            self.duration_s_sum += duration_s

    def summary(self, prefix: str) -> Dict[str, Optional[float]]:
        sample_mean = (
            self.cer_sum / self.sample_count if self.sample_count > 0 else None
        )
        corpus = (
            (self.errors / self.reference_characters) * 100
            if self.reference_characters > 0
            else (
                100.0 if self.errors > 0 else (0.0 if self.sample_count > 0 else None)
            )
        )
        duration_weighted = (
            self.duration_weighted_cer_sum / self.duration_s_sum
            if self.duration_s_sum > 0
            else None
        )
        return {
            f"{prefix}_cer_sample_count": float(self.sample_count),
            f"{prefix}_cer_errors": float(self.errors),
            f"{prefix}_reference_characters": float(self.reference_characters),
            f"{prefix}_sample_mean_cer": sample_mean,
            f"{prefix}_corpus_cer": corpus,
            f"{prefix}_duration_weighted_cer": duration_weighted,
        }


@dataclass
class ASRScoredSample:
    dataset: str
    language: str
    duration_s: float
    final_stats: WERStats
    partial_stats: Optional[WERStats]
    final_cer_stats: Optional[CERStats]
    partial_cer_stats: Optional[CERStats]


@dataclass
class ASRDatasetAggregates:
    final: WERAggregate = field(default_factory=WERAggregate)
    partial: WERAggregate = field(default_factory=WERAggregate)
    final_cer: CERAggregate = field(default_factory=CERAggregate)
    partial_cer: CERAggregate = field(default_factory=CERAggregate)


class ASRMetricAccumulator:
    """Builds ASR WER summaries across request-scoped ASR samples.

    Thread-safe: scoring runs concurrently on completion worker threads
    outside the evaluator locks, so sample collection guards its own state.
    """

    def __init__(
        self, *, normalizer: str = "english", compute_cer: bool = False
    ) -> None:
        self._lock = threading.Lock()
        self._samples: List[ASRScoredSample] = []
        self.normalizer = normalizer
        self.compute_cer = compute_cer

    def add_clip_sample(
        self,
        *,
        dataset: str,
        duration_s: float,
        final_stats: WERStats,
        partial_stats: Optional[WERStats],
        language: str = "",
        final_cer_stats: Optional[CERStats] = None,
        partial_cer_stats: Optional[CERStats] = None,
    ) -> None:
        sample = ASRScoredSample(
            dataset=dataset,
            language=language,
            duration_s=duration_s,
            final_stats=final_stats,
            partial_stats=partial_stats,
            final_cer_stats=final_cer_stats,
            partial_cer_stats=partial_cer_stats,
        )
        with self._lock:
            self._samples.append(sample)

    def get_summary(self) -> Dict[str, Optional[float]]:
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return {}

        overall = ASRDatasetAggregates()
        by_dataset: DefaultDict[str, ASRDatasetAggregates] = defaultdict(
            ASRDatasetAggregates
        )
        by_language: DefaultDict[str, ASRDatasetAggregates] = defaultdict(
            ASRDatasetAggregates
        )
        by_dataset_language: DefaultDict[tuple[str, str], ASRDatasetAggregates] = (
            defaultdict(ASRDatasetAggregates)
        )

        for sample in samples:
            aggregates = [overall, by_dataset[sample.dataset]]
            if sample.language:
                aggregates.extend(
                    [
                        by_language[sample.language],
                        by_dataset_language[(sample.dataset, sample.language)],
                    ]
                )
            for aggregate in aggregates:
                aggregate.final.add(sample.final_stats, sample.duration_s)
                if sample.partial_stats is not None:
                    aggregate.partial.add(sample.partial_stats, sample.duration_s)
                if sample.final_cer_stats is not None:
                    aggregate.final_cer.add(sample.final_cer_stats, sample.duration_s)
                if sample.partial_cer_stats is not None:
                    aggregate.partial_cer.add(
                        sample.partial_cer_stats, sample.duration_s
                    )

        summary = {}
        summary.update(_aggregate_summary(overall, "asr"))
        for dataset, aggregates in sorted(by_dataset.items()):
            slug = _metric_slug(dataset)
            summary.update(_aggregate_summary(aggregates, f"asr_dataset_{slug}"))
        for language, aggregates in sorted(by_language.items()):
            slug = _metric_slug(language)
            summary.update(_aggregate_summary(aggregates, f"asr_language_{slug}"))
        for (dataset, language), aggregates in sorted(by_dataset_language.items()):
            dataset_slug = _metric_slug(dataset)
            language_slug = _metric_slug(language)
            summary.update(
                _aggregate_summary(
                    aggregates,
                    f"asr_dataset_{dataset_slug}_language_{language_slug}",
                )
            )
        return summary


def _aggregate_summary(
    aggregates: ASRDatasetAggregates, prefix: str
) -> Dict[str, Optional[float]]:
    summary: Dict[str, Optional[float]] = {}
    summary.update(aggregates.final.summary(f"{prefix}_final"))
    summary.update(aggregates.partial.summary(f"{prefix}_partial"))
    if aggregates.final_cer.sample_count:
        summary.update(aggregates.final_cer.summary(f"{prefix}_final"))
    if aggregates.partial_cer.sample_count:
        summary.update(aggregates.partial_cer.summary(f"{prefix}_partial"))
    return summary


def _metric_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


@dataclass
class ASRRequestMetrics:
    """Request-level ASR fields persisted into request_level_metrics.jsonl."""

    audio_file: Optional[str]
    final_transcript: str
    expected_transcript: str
    partial_transcript: Optional[str]
    reference_word_timestamps: Optional[List[Dict[str, Any]]]
    transcript_snapshots: List[Dict[str, Any]]
    final_wer: Optional[float]
    partial_wer: Optional[float]
    final_cer: Optional[float]
    partial_cer: Optional[float]
    dataset: str
    language: str
    source_id: Optional[str]
    sample_id: Optional[str]
    time_to_first_visible_text: Optional[float]
    time_to_first_partial: Optional[float]
    time_to_final_transcript: Optional[float]
    interactivity: Optional[float]
    interactivity_word_count: int

    def to_request_row(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "language": self.language or None,
            "source_id": self.source_id,
            "sample_id": self.sample_id,
            "audio_file": self.audio_file,
            "time_to_first_visible_text": (
                round(self.time_to_first_visible_text, 3)
                if self.time_to_first_visible_text is not None
                else None
            ),
            "time_to_first_partial": (
                round(self.time_to_first_partial, 3)
                if self.time_to_first_partial is not None
                else None
            ),
            "time_to_final_transcript": (
                round(self.time_to_final_transcript, 3)
                if self.time_to_final_transcript is not None
                else None
            ),
            "interactivity": (
                round(self.interactivity, 3) if self.interactivity is not None else None
            ),
            "interactivity_word_count": self.interactivity_word_count,
            "partial_transcript": self.partial_transcript,
            "final_transcript": self.final_transcript,
            "expected_transcript": self.expected_transcript,
            "reference_word_timestamps": self.reference_word_timestamps,
            "transcript_snapshots": self.transcript_snapshots,
            "partial_wer": (
                round(self.partial_wer, 3) if self.partial_wer is not None else None
            ),
            "final_wer": (
                round(self.final_wer, 3) if self.final_wer is not None else None
            ),
            "partial_cer": (
                round(self.partial_cer, 3) if self.partial_cer is not None else None
            ),
            "final_cer": (
                round(self.final_cer, 3) if self.final_cer is not None else None
            ),
        }


def score_asr_request(
    *,
    request_id: int,
    channel_metrics: Dict[str, Any],
    duration_s: float,
    accumulator: ASRMetricAccumulator,
) -> ASRRequestMetrics:
    """Validate one STT response, score it, and add it to ASR aggregates."""
    final_transcript = channel_metrics.get("final_transcript")
    expected_transcript = channel_metrics.get("expected_transcript")
    if final_transcript is None or expected_transcript is None:
        raise ValueError(
            f"STT response for request {request_id} missing "
            f"final_transcript={final_transcript!r} / "
            f"expected_transcript={expected_transcript!r}."
        )

    partial_transcript = channel_metrics.get("partial_transcript")
    dataset = str(channel_metrics.get("dataset") or "unknown")
    language = str(channel_metrics.get("language") or "")
    audio_file = _optional_str(channel_metrics.get("audio_file"))
    source_id = _optional_str(channel_metrics.get("source_id"))
    sample_id = _optional_str(channel_metrics.get("sample_id"))
    reference_word_timestamps = _optional_list_of_dicts(
        channel_metrics.get("reference_word_timestamps")
    )
    transcript_snapshots = _list_of_dicts_or_empty(
        channel_metrics.get("transcript_snapshots")
    )
    time_to_first_visible_text = _optional_float(
        channel_metrics.get("time_to_first_visible_text")
    )
    time_to_first_partial = _optional_float(
        channel_metrics.get("time_to_first_partial")
    )
    time_to_final_transcript = _optional_float(
        channel_metrics.get("time_to_final_transcript")
    )
    interactivity_stats = compute_interactivity_stats(channel_metrics)
    interactivity = (
        interactivity_stats.mean_latency_ms if interactivity_stats is not None else None
    )
    interactivity_word_count = (
        interactivity_stats.word_count if interactivity_stats is not None else 0
    )

    final_stats = compute_wer_stats(
        str(expected_transcript),
        str(final_transcript),
        normalizer=accumulator.normalizer,
    )
    final_wer = final_stats.wer
    partial_stats = None
    partial_wer = None
    final_cer_stats = None
    final_cer = None
    partial_cer_stats = None
    partial_cer = None
    if accumulator.compute_cer:
        final_cer_stats = compute_cer_stats(
            str(expected_transcript),
            str(final_transcript),
            normalizer=accumulator.normalizer,
        )
        final_cer = final_cer_stats.cer
    if partial_transcript:
        partial_stats = compute_wer_stats(
            str(expected_transcript),
            str(partial_transcript),
            normalizer=accumulator.normalizer,
        )
        partial_wer = partial_stats.wer
        if accumulator.compute_cer:
            partial_cer_stats = compute_cer_stats(
                str(expected_transcript),
                str(partial_transcript),
                normalizer=accumulator.normalizer,
            )
            partial_cer = partial_cer_stats.cer
    accumulator.add_clip_sample(
        dataset=dataset,
        language=language,
        duration_s=duration_s,
        final_stats=final_stats,
        partial_stats=partial_stats,
        final_cer_stats=final_cer_stats,
        partial_cer_stats=partial_cer_stats,
    )

    return ASRRequestMetrics(
        audio_file=audio_file,
        final_transcript=str(final_transcript),
        expected_transcript=str(expected_transcript),
        partial_transcript=partial_transcript,
        reference_word_timestamps=reference_word_timestamps,
        transcript_snapshots=transcript_snapshots,
        final_wer=final_wer,
        partial_wer=partial_wer,
        final_cer=final_cer,
        partial_cer=partial_cer,
        dataset=dataset,
        language=language,
        source_id=source_id,
        sample_id=sample_id,
        time_to_first_visible_text=time_to_first_visible_text,
        time_to_first_partial=time_to_first_partial,
        time_to_final_transcript=time_to_final_transcript,
        interactivity=interactivity,
        interactivity_word_count=interactivity_word_count,
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_list_of_dicts(value: Any) -> Optional[List[Dict[str, Any]]]:
    if value is None:
        return None
    return _list_of_dicts_or_empty(value)


def _list_of_dicts_or_empty(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)
