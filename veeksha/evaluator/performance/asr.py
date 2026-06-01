"""ASR-specific scoring helpers for realtime speech-to-text benchmarks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Optional

import jiwer

from veeksha.evaluator.performance.asr_normalizer import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


@dataclass(frozen=True)
class WERStats:
    """Edit counts and WER percentage for one normalized transcript comparison."""

    errors: int
    reference_words: int
    wer: float


def compute_wer_stats(reference: str, hypothesis: str) -> WERStats:
    """WER counts and percentage using the leaderboard normalizer + jiwer."""
    ref = _normalizer(reference)
    hyp = _normalizer(hypothesis)
    output = jiwer.process_words(ref, hyp)
    errors = output.substitutions + output.deletions + output.insertions
    reference_words = output.hits + output.substitutions + output.deletions
    if reference_words == 0:
        wer = 0.0 if errors == 0 else 100.0
    else:
        wer = (errors / reference_words) * 100
    return WERStats(errors=errors, reference_words=reference_words, wer=wer)


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
            else None
        )
        duration_weighted = (
            self.duration_weighted_wer_sum / self.duration_s_sum
            if self.duration_s_sum > 0
            else None
        )
        return {
            f"{prefix}_sample_count": float(self.sample_count),
            f"{prefix}_sample_mean_wer": sample_mean,
            f"{prefix}_corpus_wer": corpus,
            f"{prefix}_duration_weighted_wer": duration_weighted,
        }


@dataclass
class ASRScoredSample:
    dataset: str
    duration_s: float
    final_stats: WERStats
    partial_stats: Optional[WERStats]


@dataclass
class ASRParentChunk:
    request_id: int
    chunk_index: int
    duration_s: float
    final_transcript: str
    partial_transcript: Optional[str]


@dataclass
class ASRParentGroup:
    dataset: str
    parent_id: str
    reference: str
    parent_duration_s: float
    expected_num_chunks: Optional[int]
    chunk_occurrences: DefaultDict[int, List[ASRParentChunk]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass
class ASRDatasetAggregates:
    final: WERAggregate = field(default_factory=WERAggregate)
    partial: WERAggregate = field(default_factory=WERAggregate)


class ASRMetricAccumulator:
    """Builds ASR WER summaries across clip- and parent-scoped samples."""

    def __init__(self) -> None:
        self._clip_samples: List[ASRScoredSample] = []
        self._parent_groups: Dict[str, ASRParentGroup] = {}

    def add_clip_sample(
        self,
        *,
        dataset: str,
        duration_s: float,
        final_stats: WERStats,
        partial_stats: Optional[WERStats],
    ) -> None:
        self._clip_samples.append(
            ASRScoredSample(
                dataset=dataset,
                duration_s=duration_s,
                final_stats=final_stats,
                partial_stats=partial_stats,
            )
        )

    def add_parent_chunk(
        self,
        *,
        dataset: str,
        parent_id: str,
        reference: str,
        parent_duration_s: float,
        expected_num_chunks: Optional[int],
        request_id: int,
        chunk_index: int,
        duration_s: float,
        final_transcript: str,
        partial_transcript: Optional[str],
    ) -> None:
        group_key = f"{dataset}:{parent_id}"
        group = self._parent_groups.get(group_key)
        if group is None:
            group = ASRParentGroup(
                dataset=dataset,
                parent_id=parent_id,
                reference=reference,
                parent_duration_s=parent_duration_s,
                expected_num_chunks=expected_num_chunks,
            )
            self._parent_groups[group_key] = group
        group.chunk_occurrences[chunk_index].append(
            ASRParentChunk(
                request_id=request_id,
                chunk_index=chunk_index,
                duration_s=duration_s,
                final_transcript=final_transcript,
                partial_transcript=partial_transcript,
            )
        )

    def _iter_samples(self) -> List[ASRScoredSample]:
        samples = list(self._clip_samples)
        for group in self._parent_groups.values():
            attempts: DefaultDict[int, Dict[int, ASRParentChunk]] = defaultdict(dict)
            for chunk_index, chunks in group.chunk_occurrences.items():
                for occurrence, chunk in enumerate(
                    sorted(chunks, key=lambda c: c.request_id)
                ):
                    attempts[occurrence][chunk_index] = chunk

            for chunks_by_index in attempts.values():
                if (
                    group.expected_num_chunks is not None
                    and len(chunks_by_index) < group.expected_num_chunks
                ):
                    continue
                chunks = [
                    chunks_by_index[index] for index in sorted(chunks_by_index)
                ]
                if not chunks:
                    continue
                duration_s = group.parent_duration_s or sum(
                    c.duration_s for c in chunks
                )
                final_hypothesis = " ".join(c.final_transcript for c in chunks)
                final_stats = compute_wer_stats(group.reference, final_hypothesis)

                partial_stats = None
                if all(c.partial_transcript for c in chunks):
                    partial_hypothesis = " ".join(
                        c.partial_transcript or "" for c in chunks
                    )
                    partial_stats = compute_wer_stats(
                        group.reference, partial_hypothesis
                    )

                samples.append(
                    ASRScoredSample(
                        dataset=group.dataset,
                        duration_s=duration_s,
                        final_stats=final_stats,
                        partial_stats=partial_stats,
                    )
                )
        return samples

    def get_summary(self) -> Dict[str, Optional[float]]:
        samples = self._iter_samples()
        if not samples:
            return {}

        overall = ASRDatasetAggregates()
        by_dataset: DefaultDict[str, ASRDatasetAggregates] = defaultdict(
            ASRDatasetAggregates
        )

        for sample in samples:
            overall.final.add(sample.final_stats, sample.duration_s)
            by_dataset[sample.dataset].final.add(sample.final_stats, sample.duration_s)
            if sample.partial_stats is not None:
                overall.partial.add(sample.partial_stats, sample.duration_s)
                by_dataset[sample.dataset].partial.add(
                    sample.partial_stats, sample.duration_s
                )

        summary = {}
        summary.update(overall.final.summary("asr_final"))
        summary.update(overall.partial.summary("asr_partial"))
        for dataset, aggregates in sorted(by_dataset.items()):
            slug = _metric_slug(dataset)
            summary.update(aggregates.final.summary(f"asr_dataset_{slug}_final"))
            summary.update(aggregates.partial.summary(f"asr_dataset_{slug}_partial"))
        return summary


def _metric_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


@dataclass
class ASRRequestMetrics:
    """Request-level ASR fields persisted into request_level_metrics.jsonl."""

    final_transcript: str
    expected_transcript: str
    partial_transcript: Optional[str]
    final_wer: Optional[float]
    partial_wer: Optional[float]
    dataset: str
    source_id: Optional[str]
    sample_id: Optional[str]
    reference_scope: str
    parent_id: Optional[str]
    chunk_index: Optional[int]
    time_to_first_partial: Optional[float]
    time_to_final_transcript: Optional[float]

    def to_request_row(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_id": self.source_id,
            "sample_id": self.sample_id,
            "reference_scope": self.reference_scope,
            "parent_id": self.parent_id,
            "chunk_index": self.chunk_index,
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
            "partial_transcript": self.partial_transcript,
            "final_transcript": self.final_transcript,
            "expected_transcript": self.expected_transcript,
            "partial_wer": (
                round(self.partial_wer, 3) if self.partial_wer is not None else None
            ),
            "final_wer": (
                round(self.final_wer, 3) if self.final_wer is not None else None
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
    source_id = _optional_str(channel_metrics.get("source_id"))
    sample_id = _optional_str(channel_metrics.get("sample_id"))
    reference_scope = str(channel_metrics.get("reference_scope") or "clip")
    parent_id = _optional_str(channel_metrics.get("parent_id"))
    chunk_index = _optional_int(channel_metrics.get("chunk_index"))
    time_to_first_partial = _optional_float(
        channel_metrics.get("time_to_first_partial")
    )
    time_to_final_transcript = _optional_float(
        channel_metrics.get("time_to_final_transcript")
    )

    final_wer: Optional[float] = None
    partial_wer: Optional[float] = None

    if reference_scope == "parent":
        parent_reference = channel_metrics.get(
            "parent_expected_transcript", expected_transcript
        )
        parent_duration_s = float(
            channel_metrics.get("parent_duration_s") or duration_s
        )
        expected_num_chunks = _optional_int(channel_metrics.get("parent_num_chunks"))
        if parent_id is None:
            raise ValueError(
                f"STT response for request {request_id} has "
                "reference_scope='parent' but no parent_id."
            )
        if chunk_index is None:
            raise ValueError(
                f"STT response for request {request_id} has "
                "reference_scope='parent' but no chunk_index."
            )
        accumulator.add_parent_chunk(
            dataset=dataset,
            parent_id=parent_id,
            reference=str(parent_reference),
            parent_duration_s=parent_duration_s,
            expected_num_chunks=expected_num_chunks,
            request_id=request_id,
            chunk_index=chunk_index,
            duration_s=duration_s,
            final_transcript=str(final_transcript),
            partial_transcript=partial_transcript,
        )
    else:
        final_stats = compute_wer_stats(str(expected_transcript), str(final_transcript))
        final_wer = final_stats.wer
        partial_stats = None
        if partial_transcript:
            partial_stats = compute_wer_stats(
                str(expected_transcript), str(partial_transcript)
            )
            partial_wer = partial_stats.wer
        accumulator.add_clip_sample(
            dataset=dataset,
            duration_s=duration_s,
            final_stats=final_stats,
            partial_stats=partial_stats,
        )

    return ASRRequestMetrics(
        final_transcript=str(final_transcript),
        expected_transcript=str(expected_transcript),
        partial_transcript=partial_transcript,
        final_wer=final_wer,
        partial_wer=partial_wer,
        dataset=dataset,
        source_id=source_id,
        sample_id=sample_id,
        reference_scope=reference_scope,
        parent_id=parent_id,
        chunk_index=chunk_index,
        time_to_first_partial=time_to_first_partial,
        time_to_final_transcript=time_to_final_transcript,
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)
