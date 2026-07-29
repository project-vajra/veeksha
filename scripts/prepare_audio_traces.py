#!/usr/bin/env python3
"""Build portable ASR traces from public datasets.

Examples:
  .venv/bin/python scripts/prepare_audio_traces.py --clips-per-dataset 128

  .venv/bin/python scripts/prepare_audio_traces.py \
    --datasets aa_voxpopuli,aa_earnings22 \
    --clips-per-dataset 128 \
    --max-duration 30

  .venv/bin/python scripts/prepare_audio_traces.py \
    --clips-per-dataset 128 \
    --without-word-timestamping

  .venv/bin/python scripts/prepare_audio_traces.py \
    --datasets ami_word_timed \
    --clips-per-dataset 128
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from huggingface_hub import hf_hub_download

########################################################################
# Constants
########################################################################

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACES_ROOT = REPO_ROOT / "traces"
AA_TRACE_OUTPUT_DIR = TRACES_ROOT / "asr" / "aa_public"
AMI_TRACE_OUTPUT_DIR = TRACES_ROOT / "asr" / "ami_word_timed"
DEFAULT_DATASETS = "aa_voxpopuli,aa_earnings22"
MANIFEST_NAME = "manifest.jsonl"
BUILD_INFO_NAME = "build_info.json"

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MIN_DURATION_S = 0.25
DEFAULT_MAX_DURATION_S = 30.0
DEFAULT_SHUFFLE_BUFFER = 0
DEFAULT_SEED = 42
TARGET_DURATION_DEVIATION_FRACTION = 0.10
AMI_MAX_GAP_S = 1.5
AMI_CACHE_DIR = REPO_ROOT / "benchmark_output" / "ami_cache"
AMI_BASE_URL = "https://groups.inf.ed.ac.uk/ami"
AMI_ANNOTATIONS_ARCHIVE = "ami_public_manual_1.6.2.zip"
AMI_ANNOTATIONS_DIR = "ami_public_manual_1.6.2"
AMI_MIX_HEADSET = "{meeting_id}.Mix-Headset.wav"
AMI_DATASET_KEY = "ami_word_timed"
AA_DATASET_KEYS = frozenset(("aa_voxpopuli", "aa_earnings22"))

NEMO_MODEL = "stt_en_fastconformer_hybrid_large_pc"
NEMO_DOCKER_IMAGE = "nvcr.io/nvidia/nemo:26.02"
NEMO_DOCKER_GPUS = "all"
NEMO_CACHE_DIR = REPO_ROOT / "benchmark_output" / "nemo_docker_cache"
NEMO_EXTRA_MOUNTS: tuple[str, ...] = ()
NEMO_ALIGN_SCRIPT = "/opt/NeMo/tools/nemo_forced_aligner/align.py"
NEMO_MANIFEST_NAME = "nemo_manifest.jsonl"
NEMO_OUTPUT_DIR_NAME = "nemo_output"
NEMO_SOURCE_AUDIO_DIR_NAME = "source_audio"

NEMO_CONTAINER_SCRIPT = r"""
set -euo pipefail

cleanup() {
  paths=("$NEMO_DOCKER_CACHE_DIR" "$NEMO_ALIGNMENT_OUTPUT_DIR")
  chown -R "$HOST_UID:$HOST_GID" "${paths[@]}" 2>/dev/null || true
}
trap cleanup EXIT

python /opt/NeMo/tools/nemo_forced_aligner/align.py "$@"
"""

DATASETS: dict[str, dict[str, str]] = {
    "aa_voxpopuli": {
        "repo": "ArtificialAnalysis/VoxPopuli-Cleaned-AA",
        "split": "test",
    },
    "aa_earnings22": {
        "repo": "ArtificialAnalysis/Earnings22-Cleaned-AA",
        "split": "test",
    },
    "ami_word_timed": {
        "repo": "AMI local word XML",
        "split": "local",
    },
}

########################################################################
# Data Models
########################################################################


@dataclass(frozen=True)
class TraceSourceOptions:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    min_duration_s: float = DEFAULT_MIN_DURATION_S
    max_duration_s: float | None = None
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class WordTiming:
    word: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class TraceClip:
    audio: np.ndarray
    transcript: str
    duration_s: float
    metadata: dict[str, Any]
    word_timestamps: list[WordTiming] | None = None
    target_duration_s: float | None = None


@dataclass(frozen=True)
class NemoItem:
    row_index: int
    audio_path: Path
    text: str


########################################################################
# CLI
########################################################################


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
        help="Requested final clips per dataset. Use 0 for all clips.",
    )
    duration_group = parser.add_mutually_exclusive_group()
    duration_group.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help=(
            "Maximum final clip duration in seconds. Timestamped clips are "
            "split on word boundaries; untimestamped longer clips are skipped."
        ),
    )
    duration_group.add_argument(
        "--target-duration",
        type=float,
        default=None,
        help=(
            "Target final clip duration in seconds. Source clips are repeated "
            "before NeMo alignment and truncated after alignment."
        ),
    )
    parser.add_argument(
        "--without-word-timestamping",
        action="store_true",
        help="Skip NeMo word timestamping.",
    )
    args = parser.parse_args()
    if args.target_duration is None and args.max_duration is None:
        args.max_duration = DEFAULT_MAX_DURATION_S
    return args


def validate_args(args: argparse.Namespace) -> list[str]:
    if args.clips_per_dataset < 0:
        raise SystemExit("--clips-per-dataset must be >= 0")
    if args.max_duration is not None and args.max_duration <= 0:
        raise SystemExit("--max-duration must be positive when set")
    if args.target_duration is not None and args.target_duration <= 0:
        raise SystemExit("--target-duration must be positive when set")
    if args.target_duration is not None and args.without_word_timestamping:
        raise SystemExit("--target-duration requires word timestamping")
    dataset_keys = selected_dataset_keys(args.datasets)
    output_dir_for_dataset_keys(dataset_keys)
    return dataset_keys


########################################################################
# Dataset Sources
########################################################################


class ASRTraceSource(ABC):
    """Base class for datasets that produce source ASR clips."""

    def __init__(self, key: str, options: TraceSourceOptions) -> None:
        self.key = key
        self.options = options
        self.spec = DATASETS[key]

    @property
    def repo(self) -> str:
        return self.spec["repo"]

    @property
    def split(self) -> str:
        return self.spec["split"]

    @abstractmethod
    def iter_clips(self) -> Iterable[TraceClip]:
        """Yield source clips before optional timestamping and chunking."""


class AATraceSource(ASRTraceSource):
    """Artificial Analysis public cleaned ASR datasets."""

    def iter_clips(self) -> Iterable[TraceClip]:
        dataset = load_dataset(self.repo, split=self.split, streaming=True)
        if self.options.shuffle_buffer > 0:
            dataset = dataset.shuffle(
                seed=self.options.seed,
                buffer_size=self.options.shuffle_buffer,
            )

        for row_index, sample in enumerate(dataset):
            transcript = clean_text(sample.get("transcript"))
            if not transcript:
                continue

            try:
                audio = decode_audio(fetch_aa_audio(sample, self.repo), self.options)
            except Exception as exc:
                print(
                    f"  WARNING: skipping {self.key} row {row_index}: {exc}",
                    file=sys.stderr,
                )
                continue

            duration_s = len(audio) / float(self.options.sample_rate)
            if duration_s < self.options.min_duration_s:
                continue

            yield self._build_clip(
                audio=audio,
                transcript=transcript,
                duration_s=duration_s,
                row={"row_index": row_index, **sample},
            )

    def _build_clip(
        self,
        *,
        audio: np.ndarray,
        transcript: str,
        duration_s: float,
        row: dict[str, Any],
    ) -> TraceClip:
        sample_id = source_id(row)
        return TraceClip(
            audio=audio,
            transcript=transcript,
            duration_s=duration_s,
            metadata={
                "source_dataset": self.repo,
                "source_split": self.split,
                "source_id": sample_id,
                "sample_id": sample_id,
            },
        )


class AMITraceSource(ASRTraceSource):
    """Local AMI audio plus AMI ``*.words.xml`` word timing annotations."""

    AUDIO_DIR = ""
    WORDS_DIR = ""
    CACHE_DIR = AMI_CACHE_DIR
    BASE_URL = AMI_BASE_URL
    AUDIO_GLOB = "{meeting_id}*.wav"

    def iter_clips(self) -> Iterable[TraceClip]:
        audio_dir = self.audio_dir
        words_dir = self.words_dir
        word_files = sorted(words_dir.rglob("*.words.xml"))
        if not word_files:
            raise SystemExit(f"No AMI *.words.xml files found under {words_dir}")

        audio_cache: dict[Path, np.ndarray] = {}
        for words_file in word_files:
            meeting_id, speaker_id = parse_ami_ids(words_file)
            try:
                audio_path = self.find_audio_file(audio_dir, meeting_id, speaker_id)
                words = parse_ami_words(words_file)
                if not words:
                    continue
                full_audio = audio_cache.get(audio_path)
                if full_audio is None:
                    full_audio = decode_audio(str(audio_path), self.options)
                    audio_cache[audio_path] = full_audio
            except Exception as exc:
                print(f"  WARNING: skipping {words_file}: {exc}", file=sys.stderr)
                continue

            for clip_index, clip_words in enumerate(
                chunk_timed_words(
                    words,
                    max_duration_s=(
                        self.options.max_duration_s or DEFAULT_MAX_DURATION_S
                    ),
                    max_gap_s=AMI_MAX_GAP_S,
                )
            ):
                clip = self._build_clip(
                    full_audio=full_audio,
                    meeting_id=meeting_id,
                    speaker_id=speaker_id,
                    clip_index=clip_index,
                    words=clip_words,
                )
                if clip is not None:
                    yield clip

    def _build_clip(
        self,
        *,
        full_audio: np.ndarray,
        meeting_id: str,
        speaker_id: str,
        clip_index: int,
        words: list[WordTiming],
    ) -> TraceClip | None:
        clip_start_ms = words[0].start_ms
        clip_end_ms = words[-1].end_ms
        duration_s = (clip_end_ms - clip_start_ms) / 1000
        if duration_s < self.options.min_duration_s:
            return None

        start_sample = ms_to_sample(clip_start_ms, self.options.sample_rate)
        end_sample = min(
            len(full_audio),
            ms_to_sample(clip_end_ms, self.options.sample_rate),
        )
        if end_sample <= start_sample:
            return None

        source = f"{meeting_id}:{speaker_id}:{clip_index}"
        relative_words = [
            WordTiming(
                word=word.word,
                start_ms=round(word.start_ms - clip_start_ms, 3),
                end_ms=round(word.end_ms - clip_start_ms, 3),
            )
            for word in words
        ]
        return TraceClip(
            audio=full_audio[start_sample:end_sample],
            transcript=" ".join(word.word for word in words),
            duration_s=duration_s,
            metadata={
                "source_dataset": "AMI",
                "source_split": "local",
                "source_id": source,
                "sample_id": source,
                "meeting_id": meeting_id,
                "speaker_id": speaker_id,
            },
            word_timestamps=relative_words,
        )

    @property
    def audio_dir(self) -> Path:
        if self.AUDIO_DIR:
            return Path(self.AUDIO_DIR)
        return self.CACHE_DIR / "wav_db"

    @property
    def words_dir(self) -> Path:
        if self.WORDS_DIR:
            return Path(self.WORDS_DIR)
        return self.ensure_annotations()

    def find_audio_file(
        self,
        audio_dir: Path,
        meeting_id: str,
        speaker_id: str,
    ) -> Path:
        if not self.AUDIO_DIR:
            return self.ensure_meeting_audio(meeting_id)

        glob_pattern = self.AUDIO_GLOB.format(
            meeting_id=meeting_id,
            speaker_id=speaker_id,
        )
        matches = sorted(audio_dir.glob(glob_pattern))
        if not matches:
            raise FileNotFoundError(
                f"No AMI audio matched {glob_pattern!r} under {audio_dir}"
            )
        return matches[0]

    def ensure_annotations(self) -> Path:
        words_dir = self.find_cached_words_dir()
        if words_dir.exists() and any(words_dir.glob("*.words.xml")):
            return words_dir

        archive_path = self.CACHE_DIR / AMI_ANNOTATIONS_ARCHIVE
        download_file(
            f"{self.BASE_URL}/AMICorpusAnnotations/{AMI_ANNOTATIONS_ARCHIVE}",
            archive_path,
        )
        safe_extract_zip(archive_path, self.CACHE_DIR)
        words_dir = self.find_cached_words_dir()
        if words_dir.exists() and any(words_dir.glob("*.words.xml")):
            return words_dir
        raise FileNotFoundError("Downloaded AMI annotations did not contain words XML")

    def find_cached_words_dir(self) -> Path:
        candidates = [
            self.CACHE_DIR / "words",
            self.CACHE_DIR / AMI_ANNOTATIONS_DIR / "words",
        ]
        for candidate in candidates:
            if candidate.exists() and any(candidate.rglob("*.words.xml")):
                return candidate
        return candidates[0]

    def ensure_meeting_audio(self, meeting_id: str) -> Path:
        wav_name = AMI_MIX_HEADSET.format(meeting_id=meeting_id)
        wav_path = self.CACHE_DIR / "wav_db" / meeting_id / "audio" / wav_name
        if wav_path.exists():
            return wav_path

        download_file(
            f"{self.BASE_URL}/AMICorpusMirror/amicorpus/"
            f"{meeting_id}/audio/{wav_name}",
            wav_path,
        )
        return wav_path


def selected_dataset_keys(raw_datasets: str) -> list[str]:
    keys = [key.strip() for key in raw_datasets.split(",") if key.strip()]
    if not keys:
        raise SystemExit("--datasets must include at least one dataset")
    unknown = [key for key in keys if key not in DATASETS]
    if unknown:
        raise SystemExit(
            f"Unknown dataset key(s): {unknown}. Supported: {', '.join(DATASETS)}"
        )
    return keys


def output_dir_for_dataset_keys(dataset_keys: Sequence[str]) -> Path:
    key_set = set(dataset_keys)
    if key_set == {AMI_DATASET_KEY}:
        return AMI_TRACE_OUTPUT_DIR
    if key_set.issubset(AA_DATASET_KEYS):
        return AA_TRACE_OUTPUT_DIR
    raise SystemExit(
        "Run AMI separately from Artificial Analysis datasets: "
        f"{AMI_DATASET_KEY} writes to {AMI_TRACE_OUTPUT_DIR}, while "
        f"{', '.join(sorted(AA_DATASET_KEYS))} write to {AA_TRACE_OUTPUT_DIR}."
    )


def supported_dataset_keys() -> list[str]:
    return list(DATASETS)


def build_trace_source(key: str, options: TraceSourceOptions) -> ASRTraceSource:
    if key in AA_DATASET_KEYS:
        return AATraceSource(key, options)
    if key == AMI_DATASET_KEY:
        return AMITraceSource(key, options)
    raise ValueError(f"Unsupported ASR trace source: {key!r}")


########################################################################
# Audio And Text Helpers
########################################################################


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def source_id(row: dict[str, Any]) -> str:
    for key in ("id", "audio_id", "file", "file_name", "path", "url"):
        value = row.get(key)
        if value:
            return str(value)
    return f"source-{row['row_index']}"


def fetch_aa_audio(row: dict[str, Any], repo: str) -> str | bytes:
    url = row.get("url")
    if not url:
        raise ValueError("row has no url column")

    url = str(url)
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()

    return hf_hub_download(repo_id=repo, repo_type="dataset", filename=url)


def download_file(url: str, target: Path) -> None:
    if target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    print(f"  Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with partial.open("wb") as out:
                shutil.copyfileobj(response, out)
        partial.replace(target)
    finally:
        if partial.exists() and not target.exists():
            partial.unlink()


def safe_extract_zip(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_root / member.filename).resolve()
            if not is_relative_to(target, output_root):
                raise ValueError(
                    f"Refusing to extract unsafe zip path: {member.filename}"
                )
        archive.extractall(output_root)


def decode_audio(source: str | bytes, options: TraceSourceOptions) -> np.ndarray:
    audio_input: str | io.BytesIO
    audio_input = io.BytesIO(source) if isinstance(source, bytes) else source

    try:
        audio, sample_rate = sf.read(audio_input, dtype="float32")
    except Exception:
        audio, sample_rate = librosa.load(audio_input, sr=None, mono=False)

    if audio.ndim > 1:
        axis = 0 if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[-1] else 1
        audio = audio.mean(axis=axis)
    return resample(np.asarray(audio, dtype=np.float32), int(sample_rate), options)


def resample(
    audio: np.ndarray,
    source_sample_rate: int,
    options: TraceSourceOptions,
) -> np.ndarray:
    if source_sample_rate == options.sample_rate:
        return audio.astype(np.float32, copy=False)

    return librosa.resample(
        audio.astype(np.float32, copy=False),
        orig_sr=source_sample_rate,
        target_sr=options.sample_rate,
    ).astype(np.float32)


def ms_to_sample(ms: float, sample_rate: int) -> int:
    return max(0, int(round(ms * sample_rate / 1000)))


########################################################################
# AMI Parsing
########################################################################


def parse_ami_ids(words_file: Path) -> tuple[str, str]:
    parts = words_file.name.split(".")
    meeting_id = parts[0]
    speaker_id = parts[1] if len(parts) > 2 else ""
    return meeting_id, speaker_id


def parse_ami_words(words_file: Path) -> list[WordTiming]:
    root = ET.parse(words_file).getroot()
    words: list[WordTiming] = []
    for elem in root.iter():
        if elem.tag.rsplit("}", 1)[-1] != "w":
            continue
        text = clean_text("".join(elem.itertext()))
        if not text:
            continue
        start_s = attr_float(elem, "starttime")
        end_s = attr_float(elem, "endtime")
        if start_s is None or end_s is None:
            raise ValueError(f"AMI word missing start/end timing in {words_file}")
        words.append(
            WordTiming(
                word=text,
                start_ms=round(start_s * 1000, 3),
                end_ms=round(end_s * 1000, 3),
            )
        )
    return sorted(words, key=lambda word: (word.start_ms, word.end_ms))


def attr_float(elem: ET.Element, suffix: str) -> float | None:
    for key, value in elem.attrib.items():
        if key.rsplit("}", 1)[-1].lower() == suffix:
            return float(value)
    return None


########################################################################
# Word Timestamping
########################################################################


class NeMoDockerWordTimestampProvider:
    def __init__(self, aligner: "NeMoDockerAligner | None" = None) -> None:
        self.aligner = aligner or NeMoDockerAligner()

    def annotate(
        self,
        *,
        dataset_key: str,
        clips: list[TraceClip],
        alignment_output_dir: Path,
        options: TraceSourceOptions,
    ) -> list[TraceClip]:
        if not clips or all(clip.word_timestamps is not None for clip in clips):
            return clips

        dataset_alignment_dir = alignment_output_dir / dataset_key
        source_audio_dir = dataset_alignment_dir / NEMO_SOURCE_AUDIO_DIR_NAME
        source_audio_dir.mkdir(parents=True, exist_ok=True)
        nemo_manifest = dataset_alignment_dir / NEMO_MANIFEST_NAME
        nemo_output_dir = dataset_alignment_dir / NEMO_OUTPUT_DIR_NAME

        items = write_nemo_inputs(
            clips=clips,
            source_audio_dir=source_audio_dir,
            nemo_manifest=nemo_manifest,
            options=options,
        )
        self.aligner.run(
            manifest_path=nemo_manifest,
            output_dir=nemo_output_dir,
            alignment_output_dir=dataset_alignment_dir,
        )
        word_timings = read_nemo_word_timings(nemo_output_dir, nemo_manifest, items)

        annotated: list[TraceClip] = []
        for index, clip in enumerate(clips):
            if clip.word_timestamps is not None:
                annotated.append(clip)
            else:
                annotated.append(replace(clip, word_timestamps=word_timings[index]))
        return annotated


class NeMoDockerAligner:
    """Runs NeMo forced alignment inside NVIDIA's Docker image."""

    def __init__(
        self,
        *,
        image: str = NEMO_DOCKER_IMAGE,
        gpus: str = NEMO_DOCKER_GPUS,
        cache_dir: Path = NEMO_CACHE_DIR,
        extra_mounts: Sequence[str] = NEMO_EXTRA_MOUNTS,
        repo_root: Path = REPO_ROOT,
    ) -> None:
        self.image = image
        self.gpus = gpus
        self.cache_dir = cache_dir.resolve()
        self.extra_mounts = tuple(mount for mount in extra_mounts if mount)
        self.repo_root = repo_root.resolve()
        self.host_uid = os.getuid()
        self.host_gid = os.getgid()

    def run(
        self,
        *,
        manifest_path: Path,
        output_dir: Path,
        alignment_output_dir: Path,
    ) -> None:
        self.prepare_cache_dir()
        subprocess.run(
            self.command(
                manifest_path=manifest_path,
                output_dir=output_dir,
                alignment_output_dir=alignment_output_dir,
            ),
            check=True,
        )

    def prepare_cache_dir(self) -> None:
        for subdir in ("home", "hf", "torch", "torchinductor", "tmp"):
            (self.cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    def command(
        self,
        *,
        manifest_path: Path,
        output_dir: Path,
        alignment_output_dir: Path,
    ) -> list[str]:
        command = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            self.gpus,
            "--ipc=host",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "-e",
            f"HOST_UID={self.host_uid}",
            "-e",
            f"HOST_GID={self.host_gid}",
            "-e",
            f"HOME={self.cache_dir / 'home'}",
            "-e",
            f"HF_HOME={self.cache_dir / 'hf'}",
            "-e",
            f"TORCH_HOME={self.cache_dir / 'torch'}",
            "-e",
            f"TORCHINDUCTOR_CACHE_DIR={self.cache_dir / 'torchinductor'}",
            "-e",
            f"TMPDIR={self.cache_dir / 'tmp'}",
            "-e",
            f"NEMO_DOCKER_CACHE_DIR={self.cache_dir}",
            "-e",
            f"NEMO_ALIGNMENT_OUTPUT_DIR={alignment_output_dir}",
            "-v",
            f"{self.repo_root}:{self.repo_root}",
            "-w",
            str(self.repo_root),
        ]

        if not is_relative_to(self.cache_dir, self.repo_root):
            command.extend(["-v", f"{self.cache_dir}:{self.cache_dir}"])
        for mount in self.extra_mounts:
            command.extend(["-v", mount])

        return command + [
            self.image,
            "bash",
            "-lc",
            NEMO_CONTAINER_SCRIPT,
            "nemo_align",
            f"manifest_filepath={manifest_path}",
            f"output_dir={output_dir}",
            'save_output_file_formats=["ctm"]',
            f"pretrained_name={NEMO_MODEL}",
        ]


def write_nemo_inputs(
    *,
    clips: list[TraceClip],
    source_audio_dir: Path,
    nemo_manifest: Path,
    options: TraceSourceOptions,
) -> list[NemoItem]:
    items: list[NemoItem] = []
    with nemo_manifest.open("w", encoding="utf-8") as manifest:
        for row_index, clip in enumerate(clips):
            audio_path = source_audio_dir / f"source_{row_index:06d}.wav"
            sf.write(
                str(audio_path),
                np.clip(clip.audio, -1.0, 1.0),
                options.sample_rate,
            )
            item = NemoItem(
                row_index=row_index,
                audio_path=audio_path,
                text=clip.transcript,
            )
            items.append(item)
            manifest.write(
                json.dumps(
                    {
                        "audio_filepath": str(audio_path.absolute()),
                        "text": clip.transcript,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return items


def read_nemo_word_timings(
    nemo_output_dir: Path,
    nemo_manifest: Path,
    items: list[NemoItem],
) -> dict[int, list[WordTiming]]:
    output_manifest = find_nemo_output_manifest(nemo_output_dir, nemo_manifest)
    nemo_rows = read_jsonl(output_manifest)
    if len(nemo_rows) != len(items):
        raise ValueError(
            f"NeMo output row count {len(nemo_rows)} does not match input "
            f"{len(items)}."
        )

    word_timings: dict[int, list[WordTiming]] = {}
    for item, nemo_row in zip(items, nemo_rows):
        ctm_path = find_word_ctm_path(nemo_row, output_manifest.parent)
        word_timings[item.row_index] = parse_word_ctm(ctm_path)
    return word_timings


def find_nemo_output_manifest(nemo_output_dir: Path, nemo_manifest: Path) -> Path:
    stem = nemo_manifest.stem
    candidates = [
        nemo_output_dir / f"{stem}_with_output_file_paths.json",
        nemo_output_dir / f"{stem}_with_ctm_paths.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(nemo_output_dir.glob(f"{stem}_with*.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find NeMo output manifest in {nemo_output_dir}")


def find_word_ctm_path(row: dict[str, Any], base_dir: Path) -> Path:
    for key, value in row.items():
        if "word" in key and "ctm" in key and value:
            path = Path(str(value))
            if path.is_absolute() or path.exists():
                return path
            return base_dir / path
    raise ValueError(f"NeMo row has no word-level CTM path: {row}")


def parse_word_ctm(path: Path) -> list[WordTiming]:
    words: list[WordTiming] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            start_s = float(parts[2])
            duration_s = float(parts[3])
            # Round to microsecond precision so second->ms conversion doesn't
            # leak float artifacts (0.1 + 0.2 -> 300.00000000000006 ms) into
            # manifests.
            words.append(
                WordTiming(
                    word=parts[4],
                    start_ms=round(start_s * 1000, 3),
                    end_ms=round((start_s + duration_s) * 1000, 3),
                )
            )
    return words


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


########################################################################
# Chunking
########################################################################


def finalize_clips(
    clips: Iterable[TraceClip],
    *,
    max_duration_s: float | None,
    target_duration_s: float | None = None,
    sample_rate: int,
) -> Iterable[TraceClip]:
    for clip in clips:
        clip_target_duration_s = clip.target_duration_s or target_duration_s
        if clip_target_duration_s is not None:
            target_clip = truncate_clip_to_target_duration(
                clip,
                target_duration_s=clip_target_duration_s,
                sample_rate=sample_rate,
            )
            if target_clip is not None:
                yield target_clip
            continue

        yield from split_or_filter_clip(
            clip,
            max_duration_s=max_duration_s,
            sample_rate=sample_rate,
        )


def repeat_clip_to_cover_target_duration(
    clip: TraceClip,
    *,
    target_duration_s: float,
    sample_rate: int,
) -> TraceClip:
    if len(clip.audio) == 0:
        return replace(
            clip,
            word_timestamps=None,
            target_duration_s=target_duration_s,
        )

    source_duration_s = len(clip.audio) / sample_rate
    repeat_count = max(1, math.ceil(target_duration_s / source_duration_s))
    audio = np.tile(clip.audio, repeat_count)
    return replace(
        clip,
        audio=audio,
        transcript=" ".join([clip.transcript] * repeat_count),
        duration_s=len(audio) / sample_rate,
        word_timestamps=None,
        target_duration_s=target_duration_s,
    )


def truncate_clip_to_target_duration(
    clip: TraceClip,
    *,
    target_duration_s: float,
    sample_rate: int,
) -> TraceClip | None:
    target_samples = max(1, int(round(target_duration_s * sample_rate)))
    end_sample = min(len(clip.audio), target_samples)
    if end_sample <= 0:
        return None

    duration_s = end_sample / sample_rate
    word_timestamps = clip.word_timestamps
    transcript = clip.transcript
    if word_timestamps is not None:
        target_end_ms = duration_s * 1000
        word_timestamps = [
            word for word in word_timestamps if word.end_ms <= target_end_ms
        ]
        transcript = " ".join(word.word for word in word_timestamps)

    return replace(
        clip,
        audio=clip.audio[:end_sample],
        transcript=transcript,
        duration_s=duration_s,
        word_timestamps=word_timestamps,
    )


def split_or_filter_clip(
    clip: TraceClip,
    *,
    max_duration_s: float | None,
    sample_rate: int,
) -> Iterable[TraceClip]:
    if max_duration_s is None or clip.duration_s <= max_duration_s:
        yield clip
        return

    if not clip.word_timestamps:
        return

    for chunk_index, words in enumerate(
        chunk_timed_words(
            clip.word_timestamps,
            max_duration_s=max_duration_s,
            max_gap_s=AMI_MAX_GAP_S,
        )
    ):
        chunk = build_timed_chunk(
            clip=clip,
            words=words,
            chunk_index=chunk_index,
            sample_rate=sample_rate,
        )
        if chunk is not None:
            yield chunk


def build_timed_chunk(
    *,
    clip: TraceClip,
    words: list[WordTiming],
    chunk_index: int,
    sample_rate: int,
) -> TraceClip | None:
    start_ms = words[0].start_ms
    end_ms = words[-1].end_ms
    start_sample = ms_to_sample(start_ms, sample_rate)
    end_sample = min(len(clip.audio), ms_to_sample(end_ms, sample_rate))
    if end_sample <= start_sample:
        return None

    relative_words = [
        WordTiming(
            word=word.word,
            start_ms=round(word.start_ms - start_ms, 3),
            end_ms=round(word.end_ms - start_ms, 3),
        )
        for word in words
    ]
    sample_id = str(clip.metadata.get("sample_id", "source"))
    metadata = {
        **clip.metadata,
        "source_parent_sample_id": sample_id,
        "chunk_index": chunk_index,
        "sample_id": f"{sample_id}:chunk-{chunk_index:05d}",
    }
    return TraceClip(
        audio=clip.audio[start_sample:end_sample],
        transcript=" ".join(word.word for word in words),
        duration_s=(end_sample - start_sample) / sample_rate,
        metadata=metadata,
        word_timestamps=relative_words,
    )


def chunk_timed_words(
    words: list[WordTiming],
    *,
    max_duration_s: float,
    max_gap_s: float,
) -> Iterable[list[WordTiming]]:
    current: list[WordTiming] = []
    max_duration_ms = max_duration_s * 1000
    max_gap_ms = max_gap_s * 1000
    for word in words:
        if current:
            clip_start_ms = current[0].start_ms
            previous_end_ms = current[-1].end_ms
            if (
                word.start_ms - previous_end_ms > max_gap_ms
                or word.end_ms - clip_start_ms > max_duration_ms
            ):
                yield current
                current = []
        current.append(word)
    if current:
        yield current


########################################################################
# Manifest Writing
########################################################################


def write_trace_dataset(
    *,
    dataset_key: str,
    clips: Iterable[TraceClip],
    output_dir: Path,
    audio_root: Path,
    rows: list[dict[str, Any]],
    start_session_id: int,
    clip_limit: int | None,
    options: TraceSourceOptions,
) -> tuple[int, int]:
    dataset_audio_dir = audio_root / dataset_key
    dataset_audio_dir.mkdir(parents=True, exist_ok=True)

    produced = 0
    session_id = start_session_id
    for clip in clips:
        if clip_limit is not None and produced >= clip_limit:
            break

        wav_path = dataset_audio_dir / f"clip_{produced:05d}.wav"
        sf.write(
            str(wav_path),
            np.clip(clip.audio, -1.0, 1.0),
            options.sample_rate,
        )
        row = {
            "session_id": session_id,
            "audio_file": wav_path.relative_to(output_dir).as_posix(),
            "expected_transcript": clip.transcript,
            "dataset": dataset_key,
            "duration_s": round(clip.duration_s, 3),
            "sample_rate": options.sample_rate,
            **clip.metadata,
        }
        if clip.word_timestamps is not None:
            row["reference_word_timestamps"] = [
                {
                    "word": word.word,
                    "start_ms": round(word.start_ms, 3),
                    "end_ms": round(word.end_ms, 3),
                }
                for word in clip.word_timestamps
            ]
        rows.append(row)
        produced += 1
        session_id += 1
        print(
            f"    [{produced}] {wav_path.relative_to(output_dir)} "
            f"({clip.duration_s:.2f}s)"
        )

    return produced, session_id


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as manifest:
        for row in rows:
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")


########################################################################
# Build Provenance
########################################################################


def git_commit(repo_root: Path = REPO_ROOT) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def resolve_source_revisions(dataset_keys: Sequence[str]) -> dict[str, str | None]:
    """Best-effort pin of upstream dataset versions for reproducibility."""
    from huggingface_hub import HfApi

    revisions: dict[str, str | None] = {}
    for key in dataset_keys:
        if key in AA_DATASET_KEYS:
            try:
                revisions[key] = HfApi().dataset_info(DATASETS[key]["repo"]).sha
            except Exception:
                revisions[key] = None
        elif key == AMI_DATASET_KEY:
            revisions[key] = AMI_ANNOTATIONS_ARCHIVE
    return revisions


def write_build_info(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    dataset_keys: Sequence[str],
    clip_count: int,
) -> Path:
    timestamping_enabled = not args.without_word_timestamping
    info = {
        "tool": "prepare_audio_traces.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "veeksha_git_commit": git_commit(),
        "argv": sys.argv[1:],
        "datasets": {key: dict(DATASETS[key]) for key in dataset_keys},
        "source_revisions": resolve_source_revisions(dataset_keys),
        "clip_count": clip_count,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "seed": DEFAULT_SEED,
        "max_duration_s": args.max_duration,
        "target_duration_s": args.target_duration,
        "word_timestamping": (
            {
                "nemo_model": NEMO_MODEL,
                "nemo_docker_image": NEMO_DOCKER_IMAGE,
                "note": "ami_word_timed uses native AMI annotations",
            }
            if timestamping_enabled
            else None
        ),
    }
    path = output_dir / BUILD_INFO_NAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    return path


########################################################################
# Orchestration
########################################################################


def main() -> None:
    args = parse_args()
    dataset_keys = validate_args(args)

    output_dir = output_dir_for_dataset_keys(dataset_keys)
    audio_root = output_dir / "audio"
    manifest_path = output_dir / MANIFEST_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    clip_limit = None if args.clips_per_dataset == 0 else args.clips_per_dataset
    target_duration_s = args.target_duration
    target_duration_deviation_s = (
        target_duration_s * TARGET_DURATION_DEVIATION_FRACTION
        if target_duration_s is not None
        else None
    )
    source_options = TraceSourceOptions(max_duration_s=args.max_duration)
    target_duration_rng = np.random.default_rng(source_options.seed)
    timestamping_enabled = not args.without_word_timestamping
    timestamp_provider = NeMoDockerWordTimestampProvider()
    rows: list[dict[str, Any]] = []
    session_id = 0

    print(f"Building ASR trace: {', '.join(dataset_keys)}")
    print(f"  Output: {output_dir}")
    print(f"  Manifest: {manifest_path}")
    if target_duration_s is not None and target_duration_deviation_s is not None:
        print(
            "  Target-duration clips will be generated with "
            f"mean {target_duration_s:g}s and standard deviation "
            f"{target_duration_deviation_s:g}s"
        )

    for dataset_key in dataset_keys:
        source = build_trace_source(dataset_key, source_options)
        print(f"  Loading {dataset_key}: {source.repo} split={source.split}")

        source_clips: list[TraceClip] = []
        collect_all = clip_limit is None or (
            timestamping_enabled
            and dataset_key == "aa_earnings22"
            and target_duration_s is None
        )
        for clip in source.iter_clips():
            source_clips.append(clip)
            if (
                not collect_all
                and clip_limit is not None
                and len(source_clips) >= clip_limit
            ):
                break

        if target_duration_s is not None:
            print(
                f"  Repeating {dataset_key} clips to cover "
                "sampled target durations before alignment"
            )
            target_durations_s = [
                float(max(DEFAULT_MIN_DURATION_S, duration_s))
                for duration_s in target_duration_rng.normal(
                    target_duration_s,
                    target_duration_deviation_s,
                    len(source_clips),
                )
            ]
            source_clips = [
                repeat_clip_to_cover_target_duration(
                    clip,
                    target_duration_s=sampled_target_duration_s,
                    sample_rate=source_options.sample_rate,
                )
                for clip, sampled_target_duration_s in zip(
                    source_clips,
                    target_durations_s,
                )
            ]

        needs_nemo = timestamping_enabled and (
            target_duration_s is not None
            or any(clip.word_timestamps is None for clip in source_clips)
        )
        if needs_nemo:
            print(f"  Aligning {dataset_key} with NeMo Docker")
            source_clips = timestamp_provider.annotate(
                dataset_key=dataset_key,
                clips=source_clips,
                alignment_output_dir=output_dir / "alignment",
                options=source_options,
            )
        elif timestamping_enabled and source_clips:
            print(f"  Using native word timestamps for {dataset_key}")

        produced, session_id = write_trace_dataset(
            dataset_key=dataset_key,
            clips=finalize_clips(
                source_clips,
                max_duration_s=source_options.max_duration_s,
                target_duration_s=target_duration_s,
                sample_rate=source_options.sample_rate,
            ),
            output_dir=output_dir,
            audio_root=audio_root,
            rows=rows,
            start_session_id=session_id,
            clip_limit=clip_limit,
            options=source_options,
        )

        if clip_limit is not None and produced < clip_limit:
            duration_filter = (
                f" matching --max-duration <= {source_options.max_duration_s:g}s"
                if source_options.max_duration_s is not None
                else (
                    f" matching --target-duration {target_duration_s:g}s"
                    if target_duration_s is not None
                    else ""
                )
            )
            raise SystemExit(
                f"{dataset_key} produced {produced} eligible clip(s)"
                f"{duration_filter}, fewer than --clips-per-dataset {clip_limit}."
            )
        if clip_limit is None and produced == 0:
            print(f"  WARNING: no clips produced for {dataset_key}", file=sys.stderr)

    write_manifest(manifest_path, rows)
    print(f"\nWrote {len(rows)} clips -> {manifest_path}")

    build_info_path = write_build_info(
        output_dir=output_dir,
        args=args,
        dataset_keys=dataset_keys,
        clip_count=len(rows),
    )
    print(f"Wrote build provenance -> {build_info_path}")


if __name__ == "__main__":
    main()
    # Some free-threaded Python builds abort in pyarrow/datasets finalizers.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
