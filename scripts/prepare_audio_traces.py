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
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import soundfile as sf

HF_DATASET = "openslr/librispeech_asr"
HF_SPLIT = "test.clean"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
AUDIO_DIR = REPO_ROOT / "traces" / "audio"
MANIFEST_PATH = REPO_ROOT / "traces" / "audio_manifest.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare audio traces from LibriSpeech test-clean.")
    parser.add_argument("--num-clips", type=int, default=64, help="Number of clips to download (default: 64)")
    parser.add_argument("--min-duration", type=float, default=20.0, help="Minimum clip duration in seconds (default: 20)")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Maximum clip duration in seconds; longer clips are trimmed (default: 30)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate in Hz (default: 16000)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from datasets import Audio, load_dataset
    except ImportError:
        print("ERROR: 'datasets' library is required.  pip install datasets", file=sys.stderr)
        sys.exit(1)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {HF_DATASET} ({HF_SPLIT}) ...")
    print(f"  Clips: {args.num_clips}  Duration: [{args.min_duration}s, {args.max_duration}s]  Sample rate: {args.sample_rate} Hz")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    entries = []
    scanned = 0

    for sample in ds:
        scanned += 1
        raw_audio = sample["audio"]
        audio_bytes = raw_audio["bytes"]
        if audio_bytes is None:
            continue

        array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if array.ndim > 1:
            array = array.mean(axis=1)
        duration_s = len(array) / sr

        if duration_s < args.min_duration:
            continue

        # Trim to max duration
        if duration_s > args.max_duration:
            array = array[: int(sr * args.max_duration)]
            duration_s = args.max_duration

        # Resample if needed
        if sr != args.sample_rate:
            try:
                import resampy
                array = resampy.resample(array, sr, args.sample_rate)
                sr = args.sample_rate
            except ImportError:
                print(f"  WARNING: clip has sr={sr}, wanted {args.sample_rate}. Install resampy to auto-resample.", file=sys.stderr)

        clip_idx = len(entries)
        wav_path = AUDIO_DIR / f"hf_clip_{clip_idx:03d}.wav"
        sf.write(str(wav_path), array, sr)

        entries.append({
            "session_id": clip_idx,
            "audio_file": str(wav_path),
            "expected_transcript": sample["text"],
        })

        print(f"  [{clip_idx + 1}/{args.num_clips}] {wav_path.name}  ({duration_s:.1f}s)")

        if len(entries) >= args.num_clips:
            break

    print(f"\nScanned {scanned} samples, kept {len(entries)} clips (>= {args.min_duration}s)")

    if len(entries) < args.num_clips:
        print(
            f"WARNING: only found {len(entries)} clips >= {args.min_duration}s "
            f"(wanted {args.num_clips}).",
            file=sys.stderr,
        )

    with open(MANIFEST_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nManifest written to {MANIFEST_PATH}  ({len(entries)} entries)")
    print(f"Audio files saved to {AUDIO_DIR}/")


if __name__ == "__main__":
    main()
