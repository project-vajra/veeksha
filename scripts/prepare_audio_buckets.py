#!/usr/bin/env python3
"""Single-pass multi-bucket audio trace generator.

Companion to ``prepare_audio_traces.py``. Streams the HF LibriSpeech
test-clean split exactly **once** and fills several fixed-duration buckets
from the same iterator, so the HF auth + connection setup cost is paid only
once instead of N times.

Each bucket produces:
  - traces/audio_<D>s/hf_clip_NNN.wav    (10 clips, each exactly D seconds)
  - traces/audio_manifest_<D>s.jsonl     (manifest pointing at the wavs)

Buckets longer than the longest natural LibriSpeech clip are filled by
concatenating successive clips and trimming to the exact target.

Usage:
    python scripts/prepare_audio_buckets.py
    python scripts/prepare_audio_buckets.py --bucket-durations 2,4,8,16,32,64
    python scripts/prepare_audio_buckets.py --num-clips 20 --bucket-durations 5,10
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HF_DATASET = "openslr/librispeech_asr"
HF_SPLIT = "test.clean"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRACES_ROOT = REPO_ROOT / "traces"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate multiple fixed-duration audio buckets in one HF stream pass.")
    p.add_argument(
        "--bucket-durations",
        type=str,
        default="2,4,8,16,32,64",
        help="Comma-separated bucket durations in seconds (default: 2,4,8,16,32,64)",
    )
    p.add_argument("--num-clips", type=int, default=10, help="Clips per bucket (default: 10)")
    p.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate Hz (default: 16000)")
    return p.parse_args()


def maybe_resample(array: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return array, sr
    try:
        import resampy

        return resampy.resample(array, sr, target_sr), target_sr
    except ImportError:
        print(
            f"  WARNING: clip has sr={sr}, wanted {target_sr}. "
            f"Install resampy to auto-resample.",
            file=sys.stderr,
        )
        return array, sr


def splice_first_n(pieces: list[tuple[np.ndarray, str]], n_samples: int) -> tuple[np.ndarray, str]:
    full = np.concatenate([a for a, _ in pieces])
    out = full[:n_samples]
    text = " ".join(t for _, t in pieces)
    return out, text


def fill_bucket(
    ds_iter,
    bucket_dur_s: float,
    num_clips: int,
    sample_rate: int,
    audio_dir: Path,
) -> tuple[list[dict], int]:
    """Pull clips from `ds_iter` until `num_clips` outputs of `bucket_dur_s` are written.

    Returns (manifest_entries, samples_consumed_from_stream).
    """
    target_samples = int(sample_rate * bucket_dur_s)
    accumulator: list[tuple[np.ndarray, str]] = []
    acc_samples = 0
    entries: list[dict] = []
    consumed = 0

    while len(entries) < num_clips:
        try:
            sample = next(ds_iter)
        except StopIteration:
            print(
                f"  WARNING: dataset exhausted with only {len(entries)}/{num_clips} clips for {bucket_dur_s:.0f}s bucket.",
                file=sys.stderr,
            )
            break
        consumed += 1

        raw_audio = sample["audio"]
        audio_bytes = raw_audio["bytes"]
        if audio_bytes is None:
            continue

        array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if array.ndim > 1:
            array = array.mean(axis=1)

        array, sr = maybe_resample(array, sr, sample_rate)
        if sr != sample_rate:
            continue  # resample failed; can't safely splice

        accumulator.append((array, sample["text"]))
        acc_samples += len(array)

        # Drain as many bucket slots as the accumulator can satisfy.
        while acc_samples >= target_samples and len(entries) < num_clips:
            out_array, out_text = splice_first_n(accumulator, target_samples)
            clip_idx = len(entries)
            wav_path = audio_dir / f"hf_clip_{clip_idx:03d}.wav"
            sf.write(str(wav_path), out_array, sample_rate)
            entries.append(
                {
                    "session_id": clip_idx,
                    "audio_file": str(wav_path),
                    "expected_transcript": out_text,
                    "duration_s": round(bucket_dur_s, 3),
                }
            )
            print(f"  [{len(entries)}/{num_clips}] {wav_path.name}  ({bucket_dur_s:.2f}s)")
            # Discard leftover so adjacent clips are independent.
            accumulator = []
            acc_samples = 0

    return entries, consumed


def main() -> None:
    args = parse_args()

    try:
        from datasets import Audio, load_dataset
    except ImportError:
        print("ERROR: 'datasets' library is required.  pip install datasets", file=sys.stderr)
        sys.exit(1)

    bucket_durs = [float(x) for x in args.bucket_durations.split(",") if x.strip()]
    if not bucket_durs:
        print("ERROR: --bucket-durations must list at least one value", file=sys.stderr)
        sys.exit(2)

    # Process longest bucket first — it eats the most stream samples and concat
    # needs the most accumulation. Short buckets fly through afterwards.
    bucket_durs_sorted = sorted(bucket_durs, reverse=True)

    print(f"Streaming {HF_DATASET} ({HF_SPLIT}) ...")
    print(f"  Buckets: {bucket_durs_sorted} s   Clips/bucket: {args.num_clips}   Sample rate: {args.sample_rate} Hz")

    ds = load_dataset(HF_DATASET, split=HF_SPLIT, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds_iter = iter(ds)

    total_consumed = 0
    for dur in bucket_durs_sorted:
        audio_dir = TRACES_ROOT / f"audio_{int(dur) if dur == int(dur) else dur}s"
        manifest_path = TRACES_ROOT / f"audio_manifest_{int(dur) if dur == int(dur) else dur}s.jsonl"
        audio_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Bucket: {dur:.0f}s ===")
        print(f"  Output: {audio_dir}/   Manifest: {manifest_path}")

        entries, consumed = fill_bucket(
            ds_iter=ds_iter,
            bucket_dur_s=dur,
            num_clips=args.num_clips,
            sample_rate=args.sample_rate,
            audio_dir=audio_dir,
        )
        total_consumed += consumed

        with open(manifest_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print(f"  Wrote {len(entries)} entries -> {manifest_path}")
        print(f"  (consumed {consumed} stream samples for this bucket)")

    print(f"\nDone. Total stream samples consumed across all buckets: {total_consumed}")


if __name__ == "__main__":
    main()
