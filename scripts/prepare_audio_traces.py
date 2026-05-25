#!/usr/bin/env python3
"""Download audio clips from HF LibriSpeech and build traces/audio_manifest.jsonl.

This script:
  1. Streams the LibriSpeech ASR dataset (test-clean) from Hugging Face Hub.
  2. Picks clips whose duration is >= MIN_DURATION_S (trimmed to MAX_DURATION_S).
  3. Saves them as WAV files under traces/audio/.
  4. Writes traces/audio_manifest.jsonl with expected transcripts.

Usage:
    python scripts/prepare_audio_traces.py                # default 64 clips
    python scripts/prepare_audio_traces.py --num-clips 128
    python scripts/prepare_audio_traces.py --min-duration 10 --max-duration 20

    # Varied-duration sweep: 50 clips spread evenly across [2s, 30s], each
    # trimmed to its assigned target. Written to a separate manifest so the
    # default dataset is preserved.
    python scripts/prepare_audio_traces.py \\
        --num-clips 50 --min-duration 2 --max-duration 30 \\
        --vary-duration \\
        --audio-subdir audio_varied \\
        --manifest-name audio_manifest_varied.jsonl
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
    parser = argparse.ArgumentParser(description="Prepare audio traces from LibriSpeech test-clean.")
    parser.add_argument("--num-clips", type=int, default=64, help="Number of clips to download (default: 64)")
    parser.add_argument("--min-duration", type=float, default=20.0, help="Minimum clip duration in seconds (default: 20)")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Maximum clip duration in seconds; longer clips are trimmed (default: 30)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate in Hz (default: 16000)")
    parser.add_argument(
        "--vary-duration",
        action="store_true",
        help="Spread target clip durations evenly between --min-duration and --max-duration "
        "(linspace) and trim each kept clip to its assigned target. Default off.",
    )
    parser.add_argument(
        "--allow-concat",
        action="store_true",
        help="When in --vary-duration mode and the next stream clip is shorter than the "
        "smallest unfilled target, concatenate successive clips until the target is met, "
        "then trim. Required for targets longer than the longest natural source clip "
        "(e.g. 32s+ in LibriSpeech test-clean).",
    )
    parser.add_argument(
        "--audio-subdir",
        type=str,
        default="audio",
        help="Subdirectory under traces/ where wav files are written (default: audio)",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="audio_manifest.jsonl",
        help="Manifest filename written under traces/ (default: audio_manifest.jsonl)",
    )
    return parser.parse_args()


def linspace(start: float, stop: float, n: int) -> list[float]:
    if n <= 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + step * i for i in range(n)]


def splice_first_n(pieces: list[tuple[np.ndarray, str]], n_samples: int) -> tuple[np.ndarray, str]:
    """Concatenate `pieces` and return the first n_samples + the joined transcript.

    `pieces` is a list of (audio_array, transcript) tuples accumulated from the
    HF stream. The caller is responsible for resetting `pieces` after the splice.
    """
    full = np.concatenate([a for a, _ in pieces])
    out = full[:n_samples]
    text = " ".join(t for _, t in pieces)
    return out, text


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


def main() -> None:
    args = parse_args()

    try:
        from datasets import Audio, load_dataset
    except ImportError:
        print("ERROR: 'datasets' library is required.  pip install datasets", file=sys.stderr)
        sys.exit(1)

    audio_dir = TRACES_ROOT / args.audio_subdir
    manifest_path = TRACES_ROOT / args.manifest_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {HF_DATASET} ({HF_SPLIT}) ...")
    if args.vary_duration:
        print(
            f"  Clips: {args.num_clips}  Duration sweep: [{args.min_duration}s, {args.max_duration}s] (linspace)  "
            f"Sample rate: {args.sample_rate} Hz"
        )
    else:
        print(
            f"  Clips: {args.num_clips}  Duration: [{args.min_duration}s, {args.max_duration}s]  "
            f"Sample rate: {args.sample_rate} Hz"
        )
    print(f"  Output: {audio_dir}/  Manifest: {manifest_path}")

    ds = load_dataset(HF_DATASET, split=HF_SPLIT, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    # In vary-duration mode we pre-compute target durations.
    # Without --allow-concat: greedily assign each clip to the LARGEST unfilled
    #   target it can satisfy (descending sort) — original trim-only behavior.
    # With --allow-concat: accumulate clips and emit them into the SMALLEST
    #   unfilled target as soon as the accumulator has enough samples
    #   (ascending sort) — needed for targets longer than the longest source clip.
    if args.vary_duration:
        targets = linspace(args.min_duration, args.max_duration, args.num_clips)
        if args.allow_concat:
            unfilled: list[tuple[int, float]] = sorted(
                list(enumerate(targets)), key=lambda it: it[1]
            )
        else:
            unfilled = sorted(list(enumerate(targets)), key=lambda it: -it[1])
        entries: list[dict | None] = [None] * args.num_clips
    else:
        unfilled = []
        entries = []

    # Concat-mode accumulator: (audio_array, transcript) pieces resampled to
    # args.sample_rate, plus a running sample count.
    accumulator: list[tuple[np.ndarray, str]] = []
    acc_samples = 0

    scanned = 0
    filled = 0

    for sample in ds:
        scanned += 1
        raw_audio = sample["audio"]
        audio_bytes = raw_audio["bytes"]
        if audio_bytes is None:
            continue

        array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if array.ndim > 1:
            array = array.mean(axis=1)

        # In concat mode, resample upfront so the accumulator stays at one rate.
        if args.vary_duration and args.allow_concat:
            array, sr = maybe_resample(array, sr, args.sample_rate)
            if sr != args.sample_rate:
                continue  # resample failed, can't safely concatenate

            accumulator.append((array, sample["text"]))
            acc_samples += len(array)

            # Drain as many targets as the accumulator can satisfy.
            while unfilled:
                orig_idx, target_dur = unfilled[0]
                target_samples = int(args.sample_rate * target_dur)
                if acc_samples < target_samples:
                    break
                out_array, out_text = splice_first_n(accumulator, target_samples)
                wav_path = audio_dir / f"hf_clip_{orig_idx:03d}.wav"
                sf.write(str(wav_path), out_array, args.sample_rate)
                entries[orig_idx] = {
                    "session_id": orig_idx,
                    "audio_file": str(wav_path),
                    "expected_transcript": out_text,
                    "duration_s": round(target_dur, 3),
                }
                unfilled.pop(0)
                accumulator = []  # discard leftover for clip independence
                acc_samples = 0
                filled += 1
                print(
                    f"  [{filled}/{args.num_clips}] {wav_path.name}  ({target_dur:.2f}s)"
                )

            if not unfilled:
                break
            continue

        duration_s = len(array) / sr

        if duration_s < args.min_duration:
            continue

        if args.vary_duration:
            # Find the largest unfilled target this clip can satisfy.
            chosen_k = None
            for k, (_orig_idx, target) in enumerate(unfilled):
                if duration_s >= target:
                    chosen_k = k
                    break
            if chosen_k is None:
                continue  # too short for every remaining target
            orig_idx, target = unfilled.pop(chosen_k)
            array = array[: int(sr * target)]
            clip_duration = target
            clip_idx = orig_idx
        else:
            # Trim to max duration
            if duration_s > args.max_duration:
                array = array[: int(sr * args.max_duration)]
                duration_s = args.max_duration
            clip_duration = duration_s
            clip_idx = len(entries)

        array, sr = maybe_resample(array, sr, args.sample_rate)

        wav_path = audio_dir / f"hf_clip_{clip_idx:03d}.wav"
        sf.write(str(wav_path), array, sr)

        entry = {
            "session_id": clip_idx,
            "audio_file": str(wav_path),
            "expected_transcript": sample["text"],
            "duration_s": round(clip_duration, 3),
        }

        if args.vary_duration:
            entries[clip_idx] = entry
        else:
            entries.append(entry)

        filled += 1
        print(f"  [{filled}/{args.num_clips}] {wav_path.name}  ({clip_duration:.2f}s)")

        if args.vary_duration:
            if not unfilled:
                break
        else:
            if len(entries) >= args.num_clips:
                break

    if args.vary_duration:
        kept_entries = [e for e in entries if e is not None]
    else:
        kept_entries = entries

    print(f"\nScanned {scanned} samples, kept {len(kept_entries)} clips")

    if len(kept_entries) < args.num_clips:
        print(
            f"WARNING: only filled {len(kept_entries)} of {args.num_clips} requested targets.",
            file=sys.stderr,
        )

    with open(manifest_path, "w") as f:
        for entry in kept_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nManifest written to {manifest_path}  ({len(kept_entries)} entries)")
    print(f"Audio files saved to {audio_dir}/")


if __name__ == "__main__":
    main()
