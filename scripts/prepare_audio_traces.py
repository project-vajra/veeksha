#!/usr/bin/env python3
"""Build portable ASR traces from the public Artificial Analysis datasets."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

TRACES_ROOT = Path(__file__).resolve().parent.parent / "traces"
DEFAULT_OUTPUT_DIR = TRACES_ROOT / "asr" / "aa_public"
DEFAULT_SAMPLE_RATE = 16000
EARNINGS22_CHUNK_SECONDS = 30.0

DATASETS: dict[str, dict[str, Any]] = {
    "aa_voxpopuli": {
        "repo": "ArtificialAnalysis/VoxPopuli-Cleaned-AA",
        "split": "test",
        "parent_scoped": False,
    },
    "aa_earnings22": {
        "repo": "ArtificialAnalysis/Earnings22-Cleaned-AA",
        "split": "test",
        "parent_scoped": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help=f"Comma-separated subset to build. Supported: {', '.join(DATASETS)}.",
    )
    parser.add_argument(
        "--clips-per-dataset",
        type=int,
        default=16,
        help="Maximum clips per dataset. Use 0 for all rows.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.25,
        help="Skip clips shorter than this many seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
        help="Skip non-parent clips longer than this many seconds.",
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help="If >0, shuffle each streaming dataset with this buffer size.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    return parser.parse_args()


def selected_dataset_keys(raw_datasets: str) -> list[str]:
    keys = [key.strip() for key in raw_datasets.split(",") if key.strip()]
    unknown = [key for key in keys if key not in DATASETS]
    if unknown:
        raise SystemExit(
            f"Unknown dataset key(s): {unknown}. Supported: {', '.join(DATASETS)}"
        )
    return keys


def load_dataset_stream(key: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("ERROR: 'datasets' is required. Install veeksha deps.") from exc

    spec = DATASETS[key]
    return load_dataset(spec["repo"], split=spec["split"], streaming=True)


def fetch_audio(row: dict[str, Any], repo: str) -> str | bytes:
    url = row.get("url")
    if not url:
        raise ValueError("row has no url column")

    url = str(url)
    if url.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, repo_type="dataset", filename=url)


def resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    import librosa

    return librosa.resample(
        audio.astype(np.float32, copy=False),
        orig_sr=source_sr,
        target_sr=target_sr,
    ).astype(np.float32)


def decode_audio(source: str | bytes, target_sr: int) -> np.ndarray:
    audio_input: str | io.BytesIO
    audio_input = io.BytesIO(source) if isinstance(source, bytes) else source

    try:
        audio, sample_rate = sf.read(audio_input, dtype="float32")
    except Exception:
        import librosa

        audio, sample_rate = librosa.load(audio_input, sr=None, mono=False)

    if audio.ndim > 1:
        axis = 0 if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[-1] else 1
        audio = audio.mean(axis=axis)
    return resample(np.asarray(audio, dtype=np.float32), int(sample_rate), target_sr)


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def source_id(row: dict[str, Any]) -> str:
    for key in ("id", "audio_id", "file", "file_name", "path", "url"):
        value = row.get(key)
        if value:
            return str(value)
    return f"source-{row['row_index']}"


def iter_audio_rows(
    key: str,
    *,
    sample_rate: int,
    min_duration: float,
    max_duration: float,
    shuffle_buffer: int,
    seed: int,
) -> Iterable[tuple[np.ndarray, str, float, dict[str, Any]]]:
    spec = DATASETS[key]
    dataset = load_dataset_stream(key)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)

    for row_index, sample in enumerate(dataset):
        transcript = clean_text(sample.get("transcript"))
        if not transcript:
            continue

        try:
            audio = decode_audio(fetch_audio(sample, spec["repo"]), sample_rate)
        except Exception as exc:
            print(f"  WARNING: skipping {key} row {row_index}: {exc}", file=sys.stderr)
            continue

        duration_s = len(audio) / float(sample_rate)
        if duration_s < min_duration:
            continue
        if not spec["parent_scoped"] and duration_s > max_duration:
            continue
        yield audio, transcript, duration_s, {"row_index": row_index, **sample}


def iter_clips(
    key: str,
    audio: np.ndarray,
    transcript: str,
    duration_s: float,
    row: dict[str, Any],
    *,
    sample_rate: int,
    min_duration: float,
) -> Iterable[tuple[np.ndarray, float, dict[str, Any]]]:
    spec = DATASETS[key]
    parent_scoped = bool(spec["parent_scoped"])
    sample_id = source_id(row)
    common_metadata = {
        "source_dataset": spec["repo"],
        "source_split": spec["split"],
        "source_id": sample_id,
        "reference_scope": "parent" if parent_scoped else "clip",
    }

    if not parent_scoped:
        yield audio, duration_s, {**common_metadata, "sample_id": sample_id}
        return

    chunk_samples = int(round(EARNINGS22_CHUNK_SECONDS * sample_rate))
    chunks: list[tuple[int, int, np.ndarray]] = []
    for chunk_index, start in enumerate(range(0, len(audio), chunk_samples)):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) / sample_rate >= min_duration:
            chunks.append((chunk_index, start, chunk))

    for chunk_index, start, chunk in chunks:
        end = start + len(chunk)
        yield (
            chunk,
            len(chunk) / float(sample_rate),
            {
                **common_metadata,
                "sample_id": f"{sample_id}:{chunk_index}",
                "parent_id": sample_id,
                "parent_duration_s": round(duration_s, 3),
                "parent_num_chunks": len(chunks),
                "chunk_index": chunk_index,
                "chunk_start_s": round(start / sample_rate, 3),
                "chunk_end_s": round(end / sample_rate, 3),
            },
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.clips_per_dataset < 0:
        raise SystemExit("--clips-per-dataset must be >= 0")
    if args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be positive")
    if args.min_duration < 0 or args.max_duration <= 0:
        raise SystemExit("--min-duration/--max-duration must be positive")
    if args.min_duration > args.max_duration:
        raise SystemExit("--min-duration must be <= --max-duration")


def should_stop_before_sample(
    key: str, produced: int, num_clips: int, clip_limit: int | None
) -> bool:
    if clip_limit is None:
        return False
    if not DATASETS[key]["parent_scoped"]:
        return produced >= clip_limit
    if produced > 0 and produced + num_clips > clip_limit:
        return True
    if produced == 0 and num_clips > clip_limit:
        print(
            "  WARNING: first parent sample expands to "
            f"{num_clips} chunks, exceeding --clips-per-dataset {clip_limit} "
            "to preserve parent-level WER.",
            file=sys.stderr,
        )
    return False


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    audio_root = output_dir / "audio"
    manifest_path = output_dir / args.manifest_name
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    dataset_keys = selected_dataset_keys(args.datasets)
    clip_limit = None if args.clips_per_dataset == 0 else args.clips_per_dataset
    rows: list[dict[str, Any]] = []
    session_id = 0

    print(f"Building public AA ASR trace: {', '.join(dataset_keys)}")
    print(f"  Output: {output_dir}")
    print(f"  Manifest: {manifest_path}")

    for key in dataset_keys:
        dataset_audio_dir = audio_root / key
        dataset_audio_dir.mkdir(parents=True, exist_ok=True)
        produced = 0

        print(f"  Loading {key}: {DATASETS[key]['repo']} split={DATASETS[key]['split']}")
        for audio, transcript, duration_s, row in iter_audio_rows(
            key,
            sample_rate=args.sample_rate,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
        ):
            clips = list(
                iter_clips(
                    key,
                    audio,
                    transcript,
                    duration_s,
                    row,
                    sample_rate=args.sample_rate,
                    min_duration=args.min_duration,
                )
            )
            if not clips:
                continue
            if should_stop_before_sample(key, produced, len(clips), clip_limit):
                break

            for clip_audio, clip_duration_s, metadata in clips:
                if (
                    clip_limit is not None
                    and not DATASETS[key]["parent_scoped"]
                    and produced >= clip_limit
                ):
                    break

                wav_path = dataset_audio_dir / f"clip_{produced:05d}.wav"
                sf.write(str(wav_path), np.clip(clip_audio, -1.0, 1.0), args.sample_rate)
                rows.append(
                    {
                        "session_id": session_id,
                        "audio_file": wav_path.relative_to(output_dir).as_posix(),
                        "expected_transcript": transcript,
                        "dataset": key,
                        "duration_s": round(clip_duration_s, 3),
                        "sample_rate": args.sample_rate,
                        **metadata,
                    }
                )
                session_id += 1
                produced += 1
                print(
                    f"    [{produced}] {wav_path.relative_to(output_dir)} "
                    f"({clip_duration_s:.2f}s)"
                )

        if produced == 0:
            print(f"  WARNING: no clips produced for {key}", file=sys.stderr)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in rows:
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} clips -> {manifest_path}")


if __name__ == "__main__":
    main()
    # Some free-threaded Python builds abort in pyarrow/datasets finalizers.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
