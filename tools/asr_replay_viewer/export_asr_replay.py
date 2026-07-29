#!/usr/bin/env python3
"""Export ASR benchmark rows into a replay-viewer JSON bundle."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from veeksha.evaluator.performance.asr_interactivity import (  # noqa: E402
    _expand_reference_words,
    _normalize_words,
    _parse_reference_words,
    _parse_transcript_snapshots,
)

DEFAULT_TRACE_MANIFESTS = (
    REPO_ROOT / "traces" / "asr" / "aa_public" / "manifest.jsonl",
    REPO_ROOT / "traces" / "asr" / "ami_word_timed" / "manifest.jsonl",
)
METRICS_DIR_NAME = "metrics"
REQUEST_METRICS_NAME = "request_level_metrics.jsonl"
SUMMARY_NAME = "summary_stats.json"
REPLAY_NAME = "asr_replay.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Benchmark run directory, metrics directory, or request_level_metrics.jsonl.",
    )
    parser.add_argument(
        "--trace-manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "Trace manifest to use for audio/reference backfill. May be passed "
            "multiple times. Defaults to known traces/asr manifests when present."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <run>/metrics/asr_replay.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = resolve_metrics_path(args.run_dir)
    metrics_dir = metrics_path.parent
    run_dir = (
        metrics_dir.parent if metrics_dir.name == METRICS_DIR_NAME else metrics_dir
    )
    output_path = args.output or metrics_dir / REPLAY_NAME

    trace_rows = load_trace_rows(args.trace_manifest or existing_default_manifests())
    request_rows = read_jsonl(metrics_path)
    summary_path = metrics_dir / SUMMARY_NAME
    summary = read_json(summary_path) if summary_path.exists() else {}

    requests = [
        build_replay_request(row, trace_rows)
        for row in sorted(request_rows, key=lambda item: item.get("request_id", 0))
        if str(row.get("audio_task") or "").lower() == "stt"
    ]

    payload = {
        "schema_version": 2,
        "run_dir": repo_relative_or_absolute(run_dir),
        "metrics_file": repo_relative_or_absolute(metrics_path),
        "summary": summary,
        "requests": requests,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(requests)} ASR replay rows -> {output_path}")


def resolve_metrics_path(path: Path) -> Path:
    candidates = []
    if path.is_file():
        candidates.append(path)
    candidates.extend(
        [
            path / REQUEST_METRICS_NAME,
            path / METRICS_DIR_NAME / REQUEST_METRICS_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find {REQUEST_METRICS_NAME} under {path}")


def existing_default_manifests() -> list[Path]:
    return [path for path in DEFAULT_TRACE_MANIFESTS if path.exists()]


def load_trace_rows(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        manifest_dir = path.resolve().parent
        for row in read_jsonl(path):
            dataset = str(row.get("dataset") or "")
            for key_field in ("sample_id", "source_id"):
                value = row.get(key_field)
                if value is not None:
                    rows[(dataset, str(value))] = attach_trace_audio_url(
                        row,
                        manifest_dir,
                    )
    return rows


def attach_trace_audio_url(row: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    row = dict(row)
    audio_file = row.get("audio_file")
    if audio_file:
        audio_path = Path(str(audio_file))
        if not audio_path.is_absolute():
            audio_path = manifest_dir / audio_path
        row["audio_url"] = repo_relative_or_absolute(audio_path)
    return row


def build_replay_request(
    row: dict[str, Any],
    trace_rows: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    dataset = str(row.get("dataset") or "")
    sample_id = optional_str(row.get("sample_id"))
    source_id = optional_str(row.get("source_id"))
    trace_row = None
    for key in (sample_id, source_id):
        if key is not None:
            trace_row = trace_rows.get((dataset, key))
            if trace_row is not None:
                break

    audio_file = optional_str(row.get("audio_file"))
    audio_url = audio_url_from_row(audio_file)
    if audio_url is None and trace_row is not None:
        audio_url = optional_str(trace_row.get("audio_url"))
        audio_file = audio_file or optional_str(trace_row.get("audio_file"))

    raw_reference_words = list_or_empty(row.get("reference_word_timestamps"))
    if not raw_reference_words and trace_row is not None:
        raw_reference_words = list_or_empty(trace_row.get("reference_word_timestamps"))

    reference_words, received_words = build_word_timings(
        raw_reference_words,
        list_or_empty(row.get("transcript_snapshots")),
    )
    latencies = [word["latency_ms"] for word in received_words]
    interactivity = (
        sum(latencies) / len(latencies) if latencies else row.get("interactivity")
    )

    request = {
        "request_id": row.get("request_id"),
        "session_id": row.get("session_id"),
        "dataset": dataset,
        "source_id": source_id,
        "sample_id": sample_id,
        "audio_file": audio_file,
        "audio_url": audio_url,
        "duration_ms": row.get("generated_audio_duration"),
        "reference_words": reference_words,
        "received_words": received_words,
        "metrics": {
            "ttfc": row.get("ttfc"),
            "time_to_first_partial": row.get("time_to_first_partial"),
            "time_to_final_transcript": row.get("time_to_final_transcript"),
            "interactivity": interactivity,
            "interactivity_word_count": len(received_words),
            "partial_wer": row.get("partial_wer"),
            "final_wer": row.get("final_wer"),
            "rtf": row.get("rtf"),
            "chunk_count": row.get("chunk_count"),
        },
    }
    request["has_replay"] = bool(reference_words and received_words)
    return request


def build_word_timings(
    raw_reference_words: list[dict[str, Any]],
    raw_snapshots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not raw_reference_words:
        return [], []

    reference_words = _expand_reference_words(
        _parse_reference_words(raw_reference_words)
    )
    reference_payload = [
        {
            "word": word.word,
            "start_ms": round(word.start_ms, 3),
            "end_ms": round(word.end_ms, 3),
        }
        for word in reference_words
    ]
    if not raw_snapshots:
        return reference_payload, []

    first_seen_ms: list[float | None] = [None] * len(reference_words)
    reference_tokens = [word.word for word in reference_words]
    for snapshot in sorted(
        _parse_transcript_snapshots(raw_snapshots),
        key=lambda item: item.elapsed_ms,
    ):
        matcher = difflib.SequenceMatcher(
            a=reference_tokens,
            b=_normalize_words(snapshot.transcript),
            autojunk=False,
        )
        for tag, ref_start, ref_end, _hyp_start, _hyp_end in matcher.get_opcodes():
            if tag != "equal":
                continue
            for ref_index in range(ref_start, ref_end):
                if (
                    first_seen_ms[ref_index] is None
                    and snapshot.elapsed_ms >= reference_words[ref_index].start_ms
                ):
                    first_seen_ms[ref_index] = snapshot.elapsed_ms

    received_words = []
    for word, seen_ms in zip(reference_words, first_seen_ms):
        if seen_ms is None:
            continue
        received_words.append(
            {
                "word": word.word,
                "time_ms": round(seen_ms, 3),
                "reference_start_ms": round(word.start_ms, 3),
                "reference_end_ms": round(word.end_ms, 3),
                "latency_ms": round(max(0.0, seen_ms - word.end_ms), 3),
            }
        )
    return reference_payload, received_words


def audio_url_from_row(audio_file: str | None) -> str | None:
    if audio_file is None:
        return None
    return repo_relative_or_absolute(Path(audio_file))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def list_or_empty(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def repo_relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    if is_relative_to(resolved, REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return str(path)


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
