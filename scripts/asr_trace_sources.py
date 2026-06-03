"""Source adapters for ASR trace preparation."""

from __future__ import annotations

import io
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from huggingface_hub import hf_hub_download

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MIN_DURATION_S = 0.25
DEFAULT_MAX_DURATION_S = 60.0
DEFAULT_SHUFFLE_BUFFER = 0
DEFAULT_SEED = 42
AMI_AUDIO_GLOB = "{meeting_id}*.wav"
AMI_MAX_GAP_S = 1.5

_DATASETS: dict[str, dict[str, Any]] = {
    "aa_voxpopuli": {
        "repo": "ArtificialAnalysis/VoxPopuli-Cleaned-AA",
        "split": "test",
        "source": "aa",
    },
    "aa_earnings22": {
        "repo": "ArtificialAnalysis/Earnings22-Cleaned-AA",
        "split": "test",
        "source": "aa",
    },
    "ami_word_timed": {
        "repo": "AMI local word XML",
        "split": "local",
        "source": "ami",
    },
}


@dataclass(frozen=True)
class TraceSourceOptions:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    min_duration_s: float = DEFAULT_MIN_DURATION_S
    max_duration_s: float | None = None
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER
    seed: int = DEFAULT_SEED
    ami_audio_dir: str = ""
    ami_words_dir: str = ""


@dataclass(frozen=True)
class TraceClip:
    audio: np.ndarray
    transcript: str
    duration_s: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TimedWord:
    word: str
    start_s: float
    end_s: float


class ASRTraceSource(ABC):
    """Base class for trace sources that yield one or more request clips."""

    def __init__(self, key: str, options: TraceSourceOptions) -> None:
        self.key = key
        self.options = options
        self.spec = _DATASETS[key]

    @property
    def repo(self) -> str:
        return str(self.spec["repo"])

    @property
    def split(self) -> str:
        return str(self.spec["split"])

    @abstractmethod
    def iter_clips(self) -> Iterable[TraceClip]:
        """Yield request-scoped ASR trace clips."""


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
            if (
                self.options.max_duration_s is not None
                and duration_s > self.options.max_duration_s
            ):
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

    def iter_clips(self) -> Iterable[TraceClip]:
        audio_dir = Path(self.options.ami_audio_dir)
        words_dir = Path(self.options.ami_words_dir)
        word_files = sorted(words_dir.rglob("*.words.xml"))
        if not word_files:
            raise SystemExit(f"No AMI *.words.xml files found under {words_dir}")

        for words_file in word_files:
            meeting_id, speaker_id = parse_ami_ids(words_file)
            audio_path = find_ami_audio_file(audio_dir, meeting_id, speaker_id)
            words = parse_ami_words(words_file)
            if not words:
                continue

            try:
                full_audio = decode_audio(str(audio_path), self.options)
            except Exception as exc:
                print(f"  WARNING: skipping {audio_path}: {exc}", file=sys.stderr)
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
        words: list[TimedWord],
    ) -> TraceClip | None:
        clip_start_s = words[0].start_s
        clip_end_s = words[-1].end_s
        duration_s = clip_end_s - clip_start_s
        if duration_s < self.options.min_duration_s:
            return None

        start_sample = max(0, int(round(clip_start_s * self.options.sample_rate)))
        end_sample = min(
            len(full_audio),
            int(round(clip_end_s * self.options.sample_rate)),
        )
        if end_sample <= start_sample:
            return None

        source = f"{meeting_id}:{speaker_id}:{clip_index}"
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
                "reference_word_timestamps": [
                    {
                        "word": word.word,
                        "start_ms": round((word.start_s - clip_start_s) * 1000, 3),
                        "end_ms": round((word.end_s - clip_start_s) * 1000, 3),
                    }
                    for word in words
                ],
            },
        )


def selected_dataset_keys(raw_datasets: str) -> list[str]:
    keys = [key.strip() for key in raw_datasets.split(",") if key.strip()]
    unknown = [key for key in keys if key not in _DATASETS]
    if unknown:
        raise SystemExit(
            f"Unknown dataset key(s): {unknown}. Supported: {', '.join(_DATASETS)}"
        )
    return keys


def supported_dataset_keys() -> list[str]:
    return list(_DATASETS)


def build_trace_source(key: str, options: TraceSourceOptions) -> ASRTraceSource:
    spec = _DATASETS[key]
    if spec["source"] == "aa":
        return AATraceSource(key, options)
    if spec["source"] == "ami":
        return AMITraceSource(key, options)
    raise ValueError(f"Unsupported ASR trace source type: {spec['source']!r}")


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


def parse_ami_ids(words_file: Path) -> tuple[str, str]:
    parts = words_file.name.split(".")
    meeting_id = parts[0]
    speaker_id = parts[1] if len(parts) > 2 else ""
    return meeting_id, speaker_id


def find_ami_audio_file(audio_dir: Path, meeting_id: str, speaker_id: str) -> Path:
    glob_pattern = AMI_AUDIO_GLOB.format(
        meeting_id=meeting_id,
        speaker_id=speaker_id,
    )
    matches = sorted(audio_dir.glob(glob_pattern))
    if not matches:
        raise FileNotFoundError(
            f"No AMI audio matched {glob_pattern!r} under {audio_dir}"
        )
    return matches[0]


def parse_ami_words(words_file: Path) -> list[TimedWord]:
    root = ET.parse(words_file).getroot()
    words: list[TimedWord] = []
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
        words.append(TimedWord(word=text, start_s=start_s, end_s=end_s))
    return sorted(words, key=lambda word: (word.start_s, word.end_s))


def chunk_timed_words(
    words: list[TimedWord],
    *,
    max_duration_s: float,
    max_gap_s: float,
) -> Iterable[list[TimedWord]]:
    current: list[TimedWord] = []
    for word in words:
        if current:
            clip_start_s = current[0].start_s
            previous_end_s = current[-1].end_s
            if (
                word.start_s - previous_end_s > max_gap_s
                or word.end_s - clip_start_s > max_duration_s
            ):
                yield current
                current = []
        current.append(word)
    if current:
        yield current


def attr_float(elem: ET.Element, suffix: str) -> float | None:
    for key, value in elem.attrib.items():
        if key.rsplit("}", 1)[-1].lower() == suffix:
            return float(value)
    return None
