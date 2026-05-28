#!/usr/bin/env python3
"""Download LibriSpeech clips and build an STT benchmark manifest.

Streams the Hugging Face LibriSpeech test-clean split exactly once and produces
``--clips-per`` clips at each requested target duration. Each clip is built by
concatenating successive source utterances and trimming to the exact target
length, so any duration is supported -- including ones longer than a single
natural utterance. Leftover audio is discarded after each clip so clips stay
independent.

Writes:
  traces/<audio-subdir>/clip_NNN.wav   one wav per produced clip
  traces/<manifest-name>               JSONL, one row per clip:
                                        {session_id, audio_file,
                                         expected_transcript, duration_s}

Usage:
  python scripts/prepare_audio_traces.py                      # 5 clips each of 2,4,8,16,32 s
  python scripts/prepare_audio_traces.py --durations 10,20 --clips-per 16
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
TRACES_ROOT = Path(__file__).resolve().parent.parent / "traces"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--durations",
        default="2,4,8,16,32",
        help="Comma-separated target clip durations in seconds (default: 2,4,8,16,32).",
    )
    p.add_argument("--clips-per", type=int, default=5, help="Clips per duration (default: 5).")
    p.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate Hz (default: 16000).")
    p.add_argument("--audio-subdir", default="audio", help="Subdir under traces/ for the wavs (default: audio).")
    p.add_argument(
        "--manifest-name",
        default="audio_manifest.jsonl",
        help="Manifest filename under traces/ (default: audio_manifest.jsonl).",
    )
    return p.parse_args()


def maybe_resample(array: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return array, sr
    try:
        import resampy

        return resampy.resample(array, sr, target_sr), target_sr
    except ImportError:
        print(
            f"  WARNING: clip sr={sr} != {target_sr}; install resampy to auto-resample.",
            file=sys.stderr,
        )
        return array, sr


def main() -> None:
    args = parse_args()

    try:
        from datasets import Audio, load_dataset
    except ImportError:
        sys.exit("ERROR: 'datasets' is required.  pip install datasets")

    durations = [float(x) for x in args.durations.split(",") if x.strip()]
    if not durations:
        sys.exit("ERROR: --durations must list at least one value")

    audio_dir = TRACES_ROOT / args.audio_subdir
    manifest_path = TRACES_ROOT / args.manifest_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {HF_DATASET} ({HF_SPLIT}) ...")
    print(f"  Durations: {durations} s   Clips/duration: {args.clips_per}   Sample rate: {args.sample_rate} Hz")
    print(f"  Output: {audio_dir}/   Manifest: {manifest_path}")

    ds = load_dataset(HF_DATASET, split=HF_SPLIT, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    def source_clips():
        """Yield (audio_array, transcript) for each decodable utterance at target SR."""
        for sample in ds:
            audio_bytes = sample["audio"]["bytes"]
            if audio_bytes is None:
                continue
            array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if array.ndim > 1:
                array = array.mean(axis=1)
            array, sr = maybe_resample(array, sr, args.sample_rate)
            if sr != args.sample_rate:
                continue  # couldn't resample; skip so splices stay single-rate
            yield array, sample["text"]

    clips = source_clips()
    entries: list[dict] = []

    # Longest durations first so the largest concatenations happen while the
    # stream is freshest; short ones fly through afterwards.
    for dur in sorted(durations, reverse=True):
        target_samples = int(args.sample_rate * dur)
        produced = 0
        while produced < args.clips_per:
            acc: list[tuple[np.ndarray, str]] = []
            acc_samples = 0
            while acc_samples < target_samples:
                nxt = next(clips, None)
                if nxt is None:
                    break
                acc.append(nxt)
                acc_samples += len(nxt[0])
            if acc_samples < target_samples:
                print(
                    f"  WARNING: stream exhausted with {produced}/{args.clips_per} clips for {dur:g}s.",
                    file=sys.stderr,
                )
                break

            out = np.concatenate([a for a, _ in acc])[:target_samples]
            text = " ".join(t for _, t in acc)
            idx = len(entries)
            wav_path = audio_dir / f"clip_{idx:03d}.wav"
            sf.write(str(wav_path), out, args.sample_rate)
            entries.append(
                {
                    "session_id": idx,
                    "audio_file": str(wav_path),
                    "expected_transcript": text,
                    "duration_s": round(dur, 3),
                }
            )
            produced += 1
            print(f"  [{dur:g}s {produced}/{args.clips_per}] {wav_path.name}")

    with open(manifest_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nWrote {len(entries)} clips -> {manifest_path}")


if __name__ == "__main__":
    main()
