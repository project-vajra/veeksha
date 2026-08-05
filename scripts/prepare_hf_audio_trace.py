#!/usr/bin/env python3
"""Materialize a pinned Hugging Face audio dataset as Veeksha traces.

The default invocation prepares every configuration used by the
``asr.indic.multidomain16.v1`` named benchmark::

    uv run python scripts/prepare_hf_audio_trace.py \
      --output-dir /scratch/$USER/veeksha-datasets/asr.indic.multidomain16.v1

Rows are kept in source order. Each configuration gets an independent,
portable ``manifest.jsonl`` and ``audio/`` directory. Audio is normalized to
mono, 16 kHz, PCM16 WAV. Generated dataset assets belong in an external data
root and must not be committed to this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

DEFAULT_REPO_ID = "ayush-shunyalabs/Indic_ASR_Eval"
DEFAULT_REVISION = "7c1d7dde6aee30d98075522bc9ec9eb26898f6ab"
DEFAULT_CONFIGS = (
    "kathbath",
    "kathbath_noisy",
    "fleurs",
    "indictts",
    "respin",
    "commonvoice",
    "gramvaani",
)
DEFAULT_SPLIT = "test"
TARGET_SAMPLE_RATE = 16_000
MANIFEST_NAME = "manifest.jsonl"
PREPARATION_METADATA_NAME = "preparation.json"

_COMMIT_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")
_SAFE_CONFIG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SOURCE_ID_COLUMNS = ("id", "audio_id", "utt_id", "utterance_id", "file_name")
INDIC_LANGUAGE_CODES = {
    "Bengali": "bn",
    "Bhojpuri": "bho",
    "Chhattisgarhi": "hne",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Magahi": "mag",
    "Maithili": "mai",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Odia": "or",
    "Punjabi": "pa",
    "Sanskrit": "sa",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Exact 40-character Hugging Face commit revision (not main/latest).",
    )
    parser.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated Hugging Face dataset configurations.",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--transcript-column", default="transcript")
    parser.add_argument("--language-column", default="language")
    parser.add_argument("--dataset-column", default="dataset")
    parser.add_argument("--duration-column", default="duration")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Materialize only the first N source rows per configuration. "
            "Any run using this option is marked noncanonical and is suitable "
            "only for smoke testing."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace already-materialized requested configuration directories.",
    )
    return parser.parse_args()


def validate_revision(revision: str) -> str:
    revision = revision.strip()
    if not _COMMIT_REVISION_RE.fullmatch(revision):
        raise ValueError(
            "--revision must be an exact 40-character Hugging Face commit hash; "
            f"got {revision!r}. Moving names such as main are not reproducible."
        )
    return revision.lower()


def parse_configs(value: str) -> tuple[str, ...]:
    configs = tuple(part.strip() for part in value.split(",") if part.strip())
    if not configs:
        raise ValueError("--configs must contain at least one configuration")
    if len(set(configs)) != len(configs):
        raise ValueError(f"--configs contains duplicates: {value!r}")
    for config in configs:
        if not _SAFE_CONFIG_RE.fullmatch(config):
            raise ValueError(f"Unsafe configuration name: {config!r}")
    return configs


def _as_nonempty_text(value: Any, *, field: str, row_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row {row_index} has no non-empty {field!r}")
    return value


def _mono_channel_first(audio: np.ndarray) -> np.ndarray:
    """Convert a Hugging Face/librosa channel-first array to mono."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=0)
    raise ValueError(f"Expected one- or two-dimensional audio, got {audio.shape}")


def _decode_file_or_bytes(source: str | bytes) -> tuple[np.ndarray, int]:
    audio_input: str | io.BytesIO
    audio_input = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        # soundfile uses frames x channels when always_2d is true.
        audio, sample_rate = sf.read(audio_input, dtype="float32", always_2d=True)
        return audio.mean(axis=1), int(sample_rate)
    except Exception:
        if isinstance(source, bytes):
            audio_input.seek(0)
        audio, sample_rate = librosa.load(audio_input, sr=None, mono=False)
        return _mono_channel_first(np.asarray(audio)), int(sample_rate)


def decode_hf_audio(value: Any) -> tuple[np.ndarray, int]:
    """Decode Hugging Face ``Audio`` values across datasets releases."""
    if isinstance(value, Mapping):
        if value.get("array") is not None:
            sample_rate = value.get("sampling_rate")
            if sample_rate is None:
                raise ValueError("Decoded audio mapping has no sampling_rate")
            audio = _mono_channel_first(np.asarray(value["array"], dtype=np.float32))
            return audio, int(sample_rate)

        encoded_bytes = value.get("bytes")
        if encoded_bytes is not None:
            if not isinstance(encoded_bytes, bytes):
                raise TypeError("Encoded audio 'bytes' must be a bytes object")
            return _decode_file_or_bytes(encoded_bytes)

        path = value.get("path")
        if path:
            return _decode_file_or_bytes(str(path))

    # datasets>=4 may expose a torchcodec-backed AudioDecoder when a caller
    # supplies a decoded dataset instead of Audio(decode=False).
    get_all_samples = getattr(value, "get_all_samples", None)
    if callable(get_all_samples):
        samples = get_all_samples()
        data = getattr(samples, "data", None)
        sample_rate = getattr(samples, "sample_rate", None)
        if data is None or sample_rate is None:
            raise ValueError("AudioDecoder returned samples without data/sample_rate")
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        return _mono_channel_first(np.asarray(data, dtype=np.float32)), int(sample_rate)

    raise TypeError(f"Unsupported Hugging Face audio value: {type(value).__name__}")


def normalize_audio(audio: np.ndarray, source_sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size:
        raise ValueError(
            f"Decoded audio must be a non-empty mono array; got {audio.shape}"
        )
    if source_sample_rate <= 0:
        raise ValueError(f"Invalid source sample rate: {source_sample_rate}")
    if not np.isfinite(audio).all():
        raise ValueError("Decoded audio contains NaN or infinite samples")
    if source_sample_rate != TARGET_SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=source_sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
        ).astype(np.float32, copy=False)
    return np.clip(audio, -1.0, 1.0)


def _source_id(row: Mapping[str, Any], config: str, row_index: int) -> str:
    for column in _SOURCE_ID_COLUMNS:
        value = row.get(column)
        if value is not None and str(value).strip():
            return f"{config}:{value}"
    return f"{config}:{row_index:06d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _replace_config_dir(temp_dir: Path, final_dir: Path, *, force: bool) -> None:
    if final_dir.exists() and not force:
        raise FileExistsError(
            f"Refusing to replace existing dataset directory {final_dir}; use --force"
        )
    if not final_dir.exists():
        temp_dir.replace(final_dir)
        return

    backup_dir = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.previous-", dir=final_dir.parent)
    )
    backup_dir.rmdir()
    final_dir.replace(backup_dir)
    try:
        temp_dir.replace(final_dir)
    except Exception:
        backup_dir.replace(final_dir)
        raise
    else:
        shutil.rmtree(backup_dir)


def prepare_config(
    *,
    repo_id: str,
    revision: str,
    config: str,
    split: str,
    output_dir: Path,
    audio_column: str = "audio",
    transcript_column: str = "transcript",
    language_column: str = "language",
    dataset_column: str = "dataset",
    duration_column: str = "duration",
    max_samples: int | None = None,
    force: bool = False,
    dataset_loader: Callable[..., Any] = load_dataset,
) -> dict[str, Any]:
    """Prepare one configuration and return its provenance metadata."""
    revision = validate_revision(revision)
    if not _SAFE_CONFIG_RE.fullmatch(config):
        raise ValueError(f"Unsafe configuration name: {config!r}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when supplied")

    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / config
    if final_dir.exists() and not force:
        raise FileExistsError(
            f"Refusing to replace existing dataset directory {final_dir}; use --force"
        )

    dataset = dataset_loader(repo_id, config, split=split, revision=revision)
    if audio_column not in dataset.column_names:
        raise ValueError(f"Dataset {config!r} has no {audio_column!r} column")
    dataset = dataset.cast_column(audio_column, Audio(decode=False))
    total_rows = len(dataset)
    selected_rows = total_rows if max_samples is None else min(max_samples, total_rows)
    if selected_rows == 0:
        raise ValueError(f"Dataset {config!r}/{split!r} is empty")

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{config}.preparing-", dir=output_dir))
    audio_dir = temp_dir / "audio"
    audio_dir.mkdir()
    manifest_path = temp_dir / MANIFEST_NAME
    total_duration_s = 0.0

    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for row_index in range(selected_rows):
                row = dataset[row_index]
                if not isinstance(row, Mapping):
                    raise TypeError(
                        f"Dataset row {row_index} is not a mapping: {type(row).__name__}"
                    )
                for required_column in (transcript_column, language_column):
                    if required_column not in row:
                        raise ValueError(
                            f"Dataset {config!r} has no {required_column!r} column"
                        )

                transcript = _as_nonempty_text(
                    row[transcript_column],
                    field=transcript_column,
                    row_index=row_index,
                )
                language = _as_nonempty_text(
                    row[language_column],
                    field=language_column,
                    row_index=row_index,
                )
                language_code = INDIC_LANGUAGE_CODES.get(language, language)
                audio, source_sample_rate = decode_hf_audio(row[audio_column])
                audio = normalize_audio(audio, source_sample_rate)
                duration_s = len(audio) / TARGET_SAMPLE_RATE
                total_duration_s += duration_s

                wav_path = audio_dir / f"clip_{row_index:06d}.wav"
                sf.write(
                    wav_path,
                    audio,
                    TARGET_SAMPLE_RATE,
                    format="WAV",
                    subtype="PCM_16",
                )

                source_duration = row.get(duration_column, duration_s)
                try:
                    source_duration = float(source_duration)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Row {row_index} has invalid {duration_column!r}: "
                        f"{source_duration!r}"
                    ) from error
                if not np.isfinite(source_duration) or source_duration <= 0:
                    raise ValueError(
                        f"Row {row_index} has invalid {duration_column!r}: "
                        f"{source_duration!r}"
                    )
                dataset_name = row.get(dataset_column, config)
                dataset_name = _as_nonempty_text(
                    dataset_name,
                    field=dataset_column,
                    row_index=row_index,
                )

                manifest_row = {
                    "session_id": row_index,
                    "audio_file": wav_path.relative_to(temp_dir).as_posix(),
                    "expected_transcript": transcript,
                    "dataset": dataset_name,
                    "language": language_code,
                    "language_name": language,
                    # duration is the source annotation; duration_s is the
                    # exact duration of the normalized file used by Veeksha.
                    "duration": source_duration,
                    "duration_s": round(duration_s, 6),
                    "source_id": _source_id(row, config, row_index),
                    "source_revision": revision,
                    "source_row_index": row_index,
                }
                manifest.write(
                    json.dumps(manifest_row, ensure_ascii=False, sort_keys=True) + "\n"
                )

        canonical = max_samples is None
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "canonical": canonical,
            "source": {
                "repo_id": repo_id,
                "revision": revision,
                "config": config,
                "split": split,
            },
            "selection": {
                "order": "source_order",
                "total_rows": total_rows,
                "materialized_rows": selected_rows,
                "max_samples": max_samples,
            },
            "audio": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "channels": 1,
                "sample_rate_hz": TARGET_SAMPLE_RATE,
                "total_duration_s": round(total_duration_s, 6),
            },
            "manifest": {
                "path": MANIFEST_NAME,
                "sha256": _sha256(manifest_path),
            },
        }
        if not canonical:
            metadata["noncanonical_reason"] = (
                "max_samples selects a prefix for smoke testing; canonical runs "
                "materialize every source row"
            )
        _write_json(temp_dir / PREPARATION_METADATA_NAME, metadata)
        _replace_config_dir(temp_dir, final_dir, force=force)
        return metadata
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _write_root_metadata(output_dir: Path) -> None:
    datasets = []
    for metadata_path in sorted(output_dir.glob(f"*/{PREPARATION_METADATA_NAME}")):
        datasets.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    _write_json(
        output_dir / PREPARATION_METADATA_NAME,
        {
            "schema_version": 1,
            "datasets": datasets,
        },
    )


def main() -> None:
    args = parse_args()
    revision = validate_revision(args.revision)
    configs = parse_configs(args.configs)
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.max_samples is not None:
        print(
            "WARNING: --max-samples creates a noncanonical smoke-test trace.",
            file=sys.stderr,
        )

    output_dir = args.output_dir.expanduser().resolve()
    for config in configs:
        print(f"Preparing {args.repo_id}/{config}@{revision} ({args.split})")
        metadata = prepare_config(
            repo_id=args.repo_id,
            revision=revision,
            config=config,
            split=args.split,
            output_dir=output_dir,
            audio_column=args.audio_column,
            transcript_column=args.transcript_column,
            language_column=args.language_column,
            dataset_column=args.dataset_column,
            duration_column=args.duration_column,
            max_samples=args.max_samples,
            force=args.force,
        )
        print(
            f"  wrote {metadata['selection']['materialized_rows']} rows to "
            f"{output_dir / config}"
        )
    _write_root_metadata(output_dir)


if __name__ == "__main__":
    main()
