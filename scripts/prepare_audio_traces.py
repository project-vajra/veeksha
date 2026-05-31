#!/usr/bin/env python3
"""Build portable STT benchmark manifests from public ASR datasets.

The generated manifest is JSONL with one row per source utterance:

    {"session_id": 0, "audio_file": "audio/voxpopuli/clip_00000.wav",
     "expected_transcript": "...", "dataset": "voxpopuli", "duration_s": 8.42}

``audio_file`` is written relative to the manifest directory so generated traces
can move with the repository/workspace.

Examples:

    # Public analog of the Artificial Analysis public datasets: VoxPopuli + Earnings22.
    python scripts/prepare_audio_traces.py

    # Fast local smoke trace from LibriSpeech.
    python scripts/prepare_audio_traces.py --preset smoke --clips-per-dataset 8

    # Larger public ASR mix.
    python scripts/prepare_audio_traces.py --preset public_asr --clips-per-dataset 64
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

TRACES_ROOT = Path(__file__).resolve().parent.parent / "traces"
DEFAULT_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_name: str
    config: str | None
    split: str
    text_columns: tuple[str, ...]
    audio_column: str = "audio"


DATASETS: dict[str, DatasetSpec] = {
    "librispeech": DatasetSpec(
        key="librispeech",
        hf_name="openslr/librispeech_asr",
        config=None,
        split="test.clean",
        text_columns=("text",),
    ),
    "voxpopuli": DatasetSpec(
        key="voxpopuli",
        hf_name="facebook/voxpopuli",
        config="en",
        split="test",
        text_columns=("normalized_text", "raw_text", "text"),
    ),
    "earnings22": DatasetSpec(
        key="earnings22",
        hf_name="distil-whisper/earnings22",
        config="chunked",
        split="test",
        text_columns=("transcription", "text"),
    ),
}

PRESETS: dict[str, tuple[str, ...]] = {
    "aa_public": ("voxpopuli", "earnings22"),
    "public_asr": ("librispeech", "voxpopuli", "earnings22"),
    "smoke": ("librispeech",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="aa_public",
        help="Dataset preset to generate. Default: aa_public.",
    )
    parser.add_argument(
        "--datasets",
        help=(
            "Comma-separated dataset keys overriding --preset. "
            f"Supported: {', '.join(sorted(DATASETS))}."
        ),
    )
    parser.add_argument(
        "--clips-per-dataset",
        type=int,
        default=16,
        help="Maximum utterances to keep per dataset. Use 0 for all matching rows.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.25,
        help="Skip utterances shorter than this many seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
        help="Skip utterances longer than this many seconds.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Output WAV sample rate in Hz. Default: {DEFAULT_SAMPLE_RATE}.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help="If >0, shuffle each streaming dataset with this buffer size.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--output-dir",
        help=(
            "Output directory. Defaults to traces/asr/<preset>. "
            "The manifest and audio/ subdirectory are written here."
        ),
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.jsonl",
        help="Manifest filename inside the output directory.",
    )
    return parser.parse_args()


def _dataset_keys(args: argparse.Namespace) -> list[str]:
    raw_keys = args.datasets.split(",") if args.datasets else PRESETS[args.preset]
    keys = [key.strip() for key in raw_keys if key.strip()]
    unknown = [key for key in keys if key not in DATASETS]
    if unknown:
        supported = ", ".join(sorted(DATASETS))
        raise SystemExit(f"Unknown dataset key(s): {unknown}. Supported: {supported}")
    return keys


def _load_dataset(spec: DatasetSpec):
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "ERROR: 'datasets' is required. Install veeksha deps."
        ) from exc

    args = [spec.hf_name]
    if spec.config is not None:
        args.append(spec.config)
    dataset = load_dataset(*args, split=spec.split, streaming=True)
    return dataset.cast_column(spec.audio_column, Audio(decode=False))


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _sample_text(sample: dict[str, Any], spec: DatasetSpec) -> str:
    for column in spec.text_columns:
        if column in sample:
            text = _normalize_text(sample[column])
            if text:
                return text
    return ""


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    try:
        import resampy

        return resampy.resample(audio, source_sr, target_sr).astype(np.float32)
    except ImportError:
        try:
            import librosa

            return librosa.resample(
                audio.astype(np.float32, copy=False),
                orig_sr=source_sr,
                target_sr=target_sr,
            ).astype(np.float32)
        except ImportError as exc:
            raise RuntimeError(
                f"Audio sample rate {source_sr} != {target_sr}; install resampy "
                "or librosa to resample."
            ) from exc


def _decode_audio_value(audio_value: Any, target_sr: int) -> tuple[np.ndarray, int]:
    if isinstance(audio_value, dict) and audio_value.get("array") is not None:
        audio = np.asarray(audio_value["array"], dtype=np.float32)
        sr = int(audio_value["sampling_rate"])
    else:
        audio_bytes = None
        audio_path = None
        if isinstance(audio_value, dict):
            audio_bytes = audio_value.get("bytes")
            audio_path = audio_value.get("path")
        elif isinstance(audio_value, (str, Path)):
            audio_path = str(audio_value)

        source = io.BytesIO(audio_bytes) if audio_bytes is not None else audio_path
        if source is None:
            raise ValueError("audio sample has neither decoded array, bytes, nor path")

        try:
            audio, sr = sf.read(source, dtype="float32")
        except Exception:
            try:
                import librosa

                audio, sr = librosa.load(source, sr=None, mono=False)
            except ImportError as exc:
                raise RuntimeError(
                    "Failed to decode audio with soundfile; install librosa for "
                    "fallback decoding."
                ) from exc

    if audio.ndim > 1:
        channel_axis = (
            0 if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[-1] else 1
        )
        audio = audio.mean(axis=channel_axis)
    audio = _resample(audio, int(sr), target_sr)
    return np.asarray(audio, dtype=np.float32), target_sr


def _iter_samples(
    spec: DatasetSpec,
    *,
    sample_rate: int,
    min_duration: float,
    max_duration: float,
    shuffle_buffer: int,
    seed: int,
) -> Iterable[tuple[np.ndarray, str, float, dict[str, Any]]]:
    dataset = _load_dataset(spec)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)

    for source_index, sample in enumerate(dataset):
        if spec.audio_column not in sample:
            continue
        text = _sample_text(sample, spec)
        if not text:
            continue
        try:
            audio, sr = _decode_audio_value(sample[spec.audio_column], sample_rate)
        except Exception as exc:
            print(
                f"  WARNING: skipping {spec.key} row {source_index}: {exc}",
                file=sys.stderr,
            )
            continue
        duration_s = len(audio) / float(sr)
        if duration_s < min_duration or duration_s > max_duration:
            continue
        yield audio, text, duration_s, {"source_index": source_index, **sample}


def _source_id(sample: dict[str, Any]) -> str | None:
    for key in ("id", "audio_id", "file", "file_name", "path"):
        value = sample.get(key)
        if value:
            return str(value)
    audio = sample.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return str(audio["path"])
    return None


def main() -> None:
    args = parse_args()
    if args.clips_per_dataset < 0:
        raise SystemExit("--clips-per-dataset must be >= 0")
    if args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be positive")
    if args.min_duration < 0 or args.max_duration <= 0:
        raise SystemExit("--min-duration/--max-duration must be positive")
    if args.min_duration > args.max_duration:
        raise SystemExit("--min-duration must be <= --max-duration")

    dataset_keys = _dataset_keys(args)
    output_dir = (
        Path(args.output_dir) if args.output_dir else TRACES_ROOT / "asr" / args.preset
    )
    audio_root = output_dir / "audio"
    manifest_path = output_dir / args.manifest_name
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    print(f"Building ASR trace preset={args.preset} datasets={','.join(dataset_keys)}")
    print(f"  Output: {output_dir}")
    print(f"  Manifest: {manifest_path}")

    rows: list[dict[str, Any]] = []
    session_id = 0
    clip_limit = None if args.clips_per_dataset == 0 else args.clips_per_dataset

    for key in dataset_keys:
        spec = DATASETS[key]
        dataset_audio_dir = audio_root / spec.key
        dataset_audio_dir.mkdir(parents=True, exist_ok=True)
        produced = 0
        print(f"  Loading {spec.key}: {spec.hf_name} split={spec.split}")

        for audio, text, duration_s, sample in _iter_samples(
            spec,
            sample_rate=args.sample_rate,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
        ):
            if clip_limit is not None and produced >= clip_limit:
                break

            wav_path = dataset_audio_dir / f"clip_{produced:05d}.wav"
            sf.write(str(wav_path), np.clip(audio, -1.0, 1.0), args.sample_rate)
            rows.append(
                {
                    "session_id": session_id,
                    "audio_file": wav_path.relative_to(output_dir).as_posix(),
                    "expected_transcript": text,
                    "dataset": spec.key,
                    "source_dataset": spec.hf_name,
                    "source_config": spec.config,
                    "source_split": spec.split,
                    "source_id": _source_id(sample),
                    "duration_s": round(duration_s, 3),
                    "sample_rate": args.sample_rate,
                }
            )
            session_id += 1
            produced += 1
            print(
                f"    [{produced}] {wav_path.relative_to(output_dir)} ({duration_s:.2f}s)"
            )

        if produced == 0:
            print(f"  WARNING: no clips produced for {spec.key}", file=sys.stderr)

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} clips -> {manifest_path}")


if __name__ == "__main__":
    main()
    # Some free-threaded Python builds abort in pyarrow/datasets finalizers.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
