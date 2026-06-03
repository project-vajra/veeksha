"""Interactivity scoring for realtime speech-to-text transcripts."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Optional

from veeksha.evaluator.performance.asr_normalizer import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass(frozen=True)
class ReferenceWord:
    word: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class TranscriptSnapshot:
    elapsed_ms: float
    transcript: str


@dataclass(frozen=True)
class InteractivityStats:
    mean_latency_ms: float
    word_count: int
    latencies_ms: list[float]


def compute_interactivity_stats(
    channel_metrics: dict[str, Any],
) -> Optional[InteractivityStats]:
    """Compute word visibility latency for one STT request.

    Missing ``reference_word_timestamps`` means the trace does not support this
    metric, so this returns ``None``. If reference timestamps are present, the
    runtime must also provide well-formed ``transcript_snapshots``.
    """
    if channel_metrics.get("reference_word_timestamps") is None:
        return None

    reference_words = _parse_reference_words(
        channel_metrics["reference_word_timestamps"]
    )
    snapshots = _parse_transcript_snapshots(
        channel_metrics.get("transcript_snapshots")
    )
    if not snapshots:
        raise ValueError(
            "reference_word_timestamps were provided, but transcript_snapshots "
            "is empty."
        )

    reference_tokens = [_single_normalized_word(word.word) for word in reference_words]
    first_seen_ms: list[Optional[float]] = [None] * len(reference_tokens)

    for snapshot in sorted(snapshots, key=lambda item: item.elapsed_ms):
        hypothesis_tokens = _normalize_words(snapshot.transcript)
        matcher = difflib.SequenceMatcher(
            a=reference_tokens,
            b=hypothesis_tokens,
            autojunk=False,
        )
        for tag, ref_start, ref_end, _hyp_start, _hyp_end in matcher.get_opcodes():
            if tag != "equal":
                continue
            for ref_index in range(ref_start, ref_end):
                if first_seen_ms[ref_index] is None:
                    first_seen_ms[ref_index] = snapshot.elapsed_ms

    latencies = [
        seen_ms - word.end_ms
        for seen_ms, word in zip(first_seen_ms, reference_words)
        if seen_ms is not None
    ]
    if not latencies:
        return None
    return InteractivityStats(
        mean_latency_ms=sum(latencies) / len(latencies),
        word_count=len(latencies),
        latencies_ms=latencies,
    )


def _parse_reference_words(value: Any) -> list[ReferenceWord]:
    rows = _expect_list(value, "reference_word_timestamps")
    if not rows:
        raise ValueError("reference_word_timestamps must not be empty.")

    words: list[ReferenceWord] = []
    for index, row in enumerate(rows):
        item = _expect_dict(row, f"reference_word_timestamps[{index}]")
        words.append(
            ReferenceWord(
                word=_expect_str(item, "word", index),
                start_ms=_expect_float(item, "start_ms", index),
                end_ms=_expect_float(item, "end_ms", index),
            )
        )
    return words


def _parse_transcript_snapshots(value: Any) -> list[TranscriptSnapshot]:
    rows = _expect_list(value, "transcript_snapshots")
    snapshots: list[TranscriptSnapshot] = []
    for index, row in enumerate(rows):
        item = _expect_dict(row, f"transcript_snapshots[{index}]")
        snapshots.append(
            TranscriptSnapshot(
                elapsed_ms=_expect_float(item, "elapsed_ms", index),
                transcript=_expect_str(item, "transcript", index),
            )
        )
    return snapshots


def _expect_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list, got {type(value).__name__}.")
    return value


def _expect_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object, got {type(value).__name__}.")
    return value


def _expect_str(item: dict[str, Any], field: str, index: int) -> str:
    value = item[field]
    if not isinstance(value, str):
        raise TypeError(f"{field} at index {index} must be str.")
    return value


def _expect_float(item: dict[str, Any], field: str, index: int) -> float:
    value = item[field]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} at index {index} must be numeric.") from exc


def _single_normalized_word(text: str) -> str:
    words = _normalize_words(text)
    if len(words) != 1:
        raise ValueError(
            "Each reference_word_timestamps entry must normalize to exactly "
            f"one word; got {text!r} -> {words!r}."
        )
    return words[0]


def _normalize_words(text: str) -> list[str]:
    return _WORD_RE.findall(_normalizer(text))
