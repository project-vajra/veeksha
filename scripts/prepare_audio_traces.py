#!/usr/bin/env python3
"""Build portable ASR traces from the public Artificial Analysis datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from align_audio_trace import align_manifest
from asr_trace_sources import (
    TraceSourceOptions,
    build_trace_source,
    selected_dataset_keys,
    supported_dataset_keys,
)

TRACES_ROOT = Path(__file__).resolve().parent.parent / "traces"
DEFAULT_OUTPUT_DIR = TRACES_ROOT / "asr" / "aa_public"
DEFAULT_DATASETS = "aa_voxpopuli,aa_earnings22"
MANIFEST_NAME = "manifest.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset_keys = supported_dataset_keys()
    parser.add_argument(
        "--datasets",
        default=DEFAULT_DATASETS,
        help=f"Comma-separated subset to build. Supported: {', '.join(dataset_keys)}.",
    )
    parser.add_argument(
        "--clips-per-dataset",
        type=int,
        default=16,
        help="Requested clips per dataset. Use 0 for all rows.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help=(
            "Only include complete clips at or below this duration in seconds. "
            "Longer clips are skipped, not trimmed."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--nemo-align-script",
        default=os.environ.get("NEMO_ALIGN_SCRIPT", ""),
        help=(
            "Path to NeMo tools/nemo_forced_aligner/align.py. When set, the "
            "manifest is rewritten with reference_word_timestamps."
        ),
    )
    parser.add_argument(
        "--ami-audio-dir",
        default="",
        help="Directory containing local AMI WAV files for ami_word_timed.",
    )
    parser.add_argument(
        "--ami-words-dir",
        default="",
        help="Directory containing AMI *.words.xml annotation files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.clips_per_dataset < 0:
        raise SystemExit("--clips-per-dataset must be >= 0")
    if args.max_duration is not None and args.max_duration <= 0:
        raise SystemExit("--max-duration must be positive when set")
    dataset_keys = selected_dataset_keys(args.datasets)
    if "ami_word_timed" in dataset_keys:
        if not args.ami_audio_dir or not args.ami_words_dir:
            raise SystemExit(
                "ami_word_timed requires --ami-audio-dir and --ami-words-dir"
            )


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    audio_root = output_dir / "audio"
    manifest_path = output_dir / MANIFEST_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    dataset_keys = selected_dataset_keys(args.datasets)
    clip_limit = None if args.clips_per_dataset == 0 else args.clips_per_dataset
    source_options = TraceSourceOptions(
        max_duration_s=args.max_duration,
        ami_audio_dir=args.ami_audio_dir,
        ami_words_dir=args.ami_words_dir,
    )
    rows: list[dict[str, Any]] = []
    session_id = 0

    print(f"Building ASR trace: {', '.join(dataset_keys)}")
    print(f"  Output: {output_dir}")
    print(f"  Manifest: {manifest_path}")

    for key in dataset_keys:
        dataset_audio_dir = audio_root / key
        dataset_audio_dir.mkdir(parents=True, exist_ok=True)
        produced = 0

        source = build_trace_source(key, source_options)
        print(f"  Loading {key}: {source.repo} split={source.split}")
        for clip in source.iter_clips():
            if clip_limit is not None and produced >= clip_limit:
                break

            wav_path = dataset_audio_dir / f"clip_{produced:05d}.wav"
            sf.write(
                str(wav_path),
                np.clip(clip.audio, -1.0, 1.0),
                source_options.sample_rate,
            )
            rows.append(
                {
                    "session_id": session_id,
                    "audio_file": wav_path.relative_to(output_dir).as_posix(),
                    "expected_transcript": clip.transcript,
                    "dataset": key,
                    "duration_s": round(clip.duration_s, 3),
                    "sample_rate": source_options.sample_rate,
                    **clip.metadata,
                }
            )
            session_id += 1
            produced += 1
            print(
                f"    [{produced}] {wav_path.relative_to(output_dir)} "
                f"({clip.duration_s:.2f}s)"
            )

        if clip_limit is not None and produced < clip_limit:
            duration_filter = (
                f" matching --max-duration <= {args.max_duration:g}s"
                if args.max_duration is not None
                else ""
            )
            raise SystemExit(
                f"{key} produced {produced} eligible clip(s){duration_filter}, "
                f"fewer than --clips-per-dataset {clip_limit}."
            )
        if clip_limit is None and produced == 0:
            print(f"  WARNING: no clips produced for {key}", file=sys.stderr)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in rows:
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} clips -> {manifest_path}")

    if args.nemo_align_script:
        alignment_output_dir = output_dir / "alignment"
        align_manifest(
            manifest=manifest_path,
            output_manifest=manifest_path,
            alignment_output_dir=alignment_output_dir,
            nemo_align_script=args.nemo_align_script,
        )
        print(f"Wrote word-timestamped manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
    # Some free-threaded Python builds abort in pyarrow/datasets finalizers.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
