"""Position-resolved long-form TTS quality scoring.

Implements the quality-ladder Tier-2 drift metrics for a single long-form
waveform:

- WER(t): non-overlapping 28 s Whisper chunks (never >30 s per chunk), the
  exact Seed-TTS-Eval text normalization, ONE global jiwer alignment of the
  concatenated transcript against the reference, and per-chunk approximate
  WER via global-alignment attribution (see ``attribute_alignment``).
- UTMOS(t): balacoon TorchScript UTMOS on 10 s chunks at 16 kHz, per-bucket
  mean and min.
- Repetition/omission: per-chunk insertion vs deletion split, duplicated
  n-gram counts, zlib compression-ratio loop flag (Whisper convention).
- Energy: RMS (dBFS) and silence fraction per fixed bin, model-free.
- SIM(t) (optional): WavLM-SV cosine on non-overlapping 3 s windows vs a
  prompt (or audio-head) anchor embedding; skipped with a note when the
  checkpoint is not available.

Chunk-to-reference alignment method: instead of re-aligning every chunk
against a searched reference region, the concatenated transcript is aligned
against the full reference ONCE with jiwer, then each alignment op is
attributed to chunks by its hypothesis-word range (each chunk owns a known
half-open span of concatenated hypothesis words). Substitution/hit/insertion
ops split proportionally across chunk boundaries; deletion ops (empty
hypothesis range) are attributed to the chunk containing their hypothesis
position, split duration-proportionally with any empty (silent) chunks at
that position so fully-silent chunks still receive their deletion mass.
Per-chunk counts are floats and sum exactly to the global counts.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import numpy as np
import scipy.signal

from veeksha.config.score_tts_longform import (
    LongformAsrConfig,
    LongformSimConfig,
    LongformUtmosConfig,
    ScoreTtsLongformConfig,
)
from veeksha.logger import init_logger
from veeksha.verification.audio import (
    load_utmos_jit_model,
    normalize_text,
    predict_utmos_f32_16k,
)

logger = init_logger(__name__)

MODEL_SAMPLE_RATE = 16000
_ENERGY_FRAME_SECONDS = 0.03
_DBFS_FLOOR = -100.0
_PROGRESS_LOG_EVERY = 25


# ---------------------------------------------------------------------------
# Audio access
# ---------------------------------------------------------------------------


class AudioSource:
    """Random-access mono float32 view over a WAV file or raw int16 PCM.

    WAV files are fully loaded (mono-averaged); raw PCM is memory-mapped so
    multi-hour int16 captures never materialize as float32 at once.
    """

    def __init__(self, path: Path, pcm_sample_rate: int):
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        self.path = path
        if path.suffix.lower() == ".wav":
            import soundfile as sf

            data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
            self._data = np.mean(data, axis=1).astype(np.float32)
            self._scale = 1.0
            self.sample_rate = int(sample_rate)
        else:
            self._data = np.memmap(str(path), dtype=np.int16, mode="r")
            self._scale = 1.0 / 32768.0
            self.sample_rate = pcm_sample_rate
        if self.num_samples == 0:
            raise ValueError(f"Audio file is empty: {path}")

    @property
    def num_samples(self) -> int:
        return int(len(self._data))

    @property
    def duration_s(self) -> float:
        return self.num_samples / self.sample_rate

    def slice_f32(self, start_sample: int, end_sample: int) -> np.ndarray:
        if not 0 <= start_sample <= end_sample <= self.num_samples:
            raise ValueError(
                f"Bad slice [{start_sample}, {end_sample}) for "
                f"{self.num_samples} samples"
            )
        chunk = np.asarray(self._data[start_sample:end_sample], dtype=np.float32)
        if self._scale != 1.0:
            chunk = chunk * self._scale
        return chunk

    def slice_f32_16k(self, start_sample: int, end_sample: int) -> np.ndarray:
        return resample_to_16k(
            self.slice_f32(start_sample, end_sample), self.sample_rate
        )


def resample_to_16k(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample to 16 kHz with scipy.signal.resample (seed protocol choice)."""
    if sample_rate == MODEL_SAMPLE_RATE:
        return wav.astype(np.float32)
    if len(wav) == 0:
        return np.zeros(0, dtype=np.float32)
    target_len = max(1, int(round(len(wav) * MODEL_SAMPLE_RATE / sample_rate)))
    return np.asarray(scipy.signal.resample(wav, target_len), dtype=np.float32)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkSpan:
    index: int
    start_s: float
    end_s: float
    start_sample: int
    end_sample: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def make_chunk_spans(
    num_samples: int,
    sample_rate: int,
    chunk_seconds: float,
    min_tail_seconds: float = 0.0,
) -> list[ChunkSpan]:
    """Tile [0, num_samples) with non-overlapping chunks of chunk_seconds.

    The trailing partial chunk is kept only if it is at least
    min_tail_seconds long.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    chunk_samples = int(round(chunk_seconds * sample_rate))
    if chunk_samples <= 0:
        raise ValueError("chunk_seconds too small for sample_rate")

    spans: list[ChunkSpan] = []
    start = 0
    while start < num_samples:
        end = min(start + chunk_samples, num_samples)
        duration = (end - start) / sample_rate
        if end - start < chunk_samples and duration < min_tail_seconds:
            break
        spans.append(
            ChunkSpan(
                index=len(spans),
                start_s=start / sample_rate,
                end_s=end / sample_rate,
                start_sample=start,
                end_sample=end,
            )
        )
        start = end
    return spans


# ---------------------------------------------------------------------------
# Pluggable scoring backends (injectable for tests)
# ---------------------------------------------------------------------------


class ChunkTranscriber(Protocol):
    def transcribe(self, wav_16k: np.ndarray) -> str:
        """Transcribe 16 kHz mono float32 audio to text."""
        ...


class UtmosScorer(Protocol):
    def score(self, wav_16k: np.ndarray) -> Optional[float]:
        """Score 16 kHz mono float32 audio; None if no finite score."""
        ...


class SimEmbedder(Protocol):
    def embed(self, wav_16k: np.ndarray) -> np.ndarray:
        """Embed 16 kHz mono float32 audio into a speaker vector."""
        ...


class FasterWhisperChunkTranscriber:
    """Seed-protocol-adjacent Whisper transcription via faster-whisper.

    Seed-TTS-Eval uses HF openai/whisper-large-v3 with greedy decoding.
    This transcriber matches the decoding settings (greedy: beam_size=1,
    temperature=0.0), pins the language, disables VAD and previous-text
    conditioning so chunks are independent. The remaining deviation from
    seed is the backend (CTranslate2 weights vs HF transformers).
    """

    def __init__(self, config: LongformAsrConfig):
        whisper_model_class = import_module("faster_whisper").WhisperModel
        self._model = whisper_model_class(
            config.model,
            device=config.device,
            compute_type=config.compute_type,
        )
        self._language = config.language

    def transcribe(self, wav_16k: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            wav_16k,
            language=self._language,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class BalacoonUtmosScorer:
    """UTMOS via the balacoon TorchScript model (shared with verification)."""

    def __init__(self, config: LongformUtmosConfig):
        self._config = config

    def score(self, wav_16k: np.ndarray) -> Optional[float]:
        return predict_utmos_f32_16k(
            wav_16k,
            self._config.hf_repo,
            self._config.jit_file,
            self._config.device,
        )


class TorchScriptSimEmbedder:
    """WavLM-SV speaker embedder loaded from a TorchScript export."""

    def __init__(self, checkpoint_path: Path, device: str):
        torch: Any = import_module("torch")

        self._torch = torch
        self._model = torch.jit.load(str(checkpoint_path), map_location=device)
        self._model.eval()
        self._device = device

    def embed(self, wav_16k: np.ndarray) -> np.ndarray:
        wav = np.ascontiguousarray(wav_16k, dtype=np.float32)
        model_input = self._torch.from_numpy(wav).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            output = self._model(model_input)
        return output.reshape(-1).cpu().numpy().astype(np.float32)


def build_transcriber(
    config: LongformAsrConfig,
) -> tuple[Optional[ChunkTranscriber], Optional[str]]:
    """Build the Whisper transcriber, or (None, note) when unavailable."""
    try:
        return FasterWhisperChunkTranscriber(config), None
    except Exception as exc:
        return None, (
            f"WER skipped: faster-whisper model '{config.model}' unavailable "
            f"({type(exc).__name__}: {exc})"
        )


def build_utmos_scorer(
    config: LongformUtmosConfig,
) -> tuple[Optional[UtmosScorer], Optional[str]]:
    """Build the UTMOS scorer, or (None, note) when unavailable."""
    model = load_utmos_jit_model(config.hf_repo, config.jit_file, config.device)
    if model is None:
        return None, (
            f"UTMOS skipped: TorchScript model {config.hf_repo}/{config.jit_file} "
            "unavailable (requires torch + huggingface_hub + model access)"
        )
    return BalacoonUtmosScorer(config), None


def build_sim_embedder(
    config: LongformSimConfig,
) -> tuple[Optional[SimEmbedder], Optional[str]]:
    """Build the WavLM-SV embedder, or (None, note) when unavailable."""
    if not config.checkpoint_path:
        return None, (
            "SIM skipped: checkpoint not available (set --sim.checkpoint_path to "
            "a TorchScript export of UniSpeech wavlm_large_finetune.pth; see "
            "--help for the download source)"
        )
    checkpoint = Path(config.checkpoint_path)
    if not checkpoint.exists():
        return None, f"SIM skipped: checkpoint not available at {checkpoint}"
    try:
        return TorchScriptSimEmbedder(checkpoint, config.device), None
    except Exception as exc:
        return None, (
            f"SIM skipped: failed to load {checkpoint} as TorchScript "
            f"({type(exc).__name__}: {exc}). Raw UniSpeech state-dict "
            "checkpoints need the UniSpeech model code; export to TorchScript "
            "first."
        )


# ---------------------------------------------------------------------------
# Repetition / loop detectors
# ---------------------------------------------------------------------------


def count_duplicated_ngrams(words: Sequence[str], ngram_size: int) -> int:
    """Count duplicated n-gram occurrences: sum of (count - 1) over n-grams.

    WhisperX-style loop detector; 0 for transcripts shorter than n words.
    """
    if len(words) < ngram_size:
        return 0
    ngrams = Counter(
        tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)
    )
    return sum(count - 1 for count in ngrams.values() if count > 1)


def compression_ratio(text: str) -> float:
    """zlib compression ratio of the transcript (Whisper loop heuristic)."""
    if not text:
        return 0.0
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw))


# ---------------------------------------------------------------------------
# Global alignment + per-chunk attribution
# ---------------------------------------------------------------------------


@dataclass
class ChunkAlignmentCounts:
    """Float alignment counts attributed to one chunk (sum to global)."""

    ref_words: float = 0.0
    hits: float = 0.0
    substitutions: float = 0.0
    deletions: float = 0.0
    insertions: float = 0.0

    @property
    def errors(self) -> float:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> Optional[float]:
        if self.ref_words <= 0.0:
            return None
        return self.errors / self.ref_words


@dataclass(frozen=True)
class GlobalWerStats:
    wer: float
    hits: int
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int
    hyp_words: int


def align_chunks_to_reference(
    reference_words: Sequence[str],
    chunk_words: Sequence[Sequence[str]],
    chunk_durations: Sequence[float],
) -> tuple[GlobalWerStats, list[ChunkAlignmentCounts]]:
    """Align concatenated chunk transcripts to the reference once with jiwer,
    then attribute ops to chunks by hypothesis-word position.

    Deletion ops carry no hypothesis words; a deletion at hypothesis position
    p is split duration-proportionally among the chunk whose hypothesis span
    contains p and any empty chunks located exactly at p, so silent chunks
    receive their share of omission mass instead of leaking it to neighbors.
    """
    import jiwer

    if not reference_words:
        raise ValueError("Reference text has no words after normalization")
    if len(chunk_words) != len(chunk_durations):
        raise ValueError("chunk_words and chunk_durations length mismatch")
    if not chunk_words:
        raise ValueError("At least one transcript chunk is required")

    hyp_offsets = [0]
    for words in chunk_words:
        hyp_offsets.append(hyp_offsets[-1] + len(words))
    hyp_words = [word for words in chunk_words for word in words]

    output = jiwer.process_words(" ".join(reference_words), " ".join(hyp_words))
    global_stats = GlobalWerStats(
        wer=float(output.wer),
        hits=int(output.hits),
        substitutions=int(output.substitutions),
        deletions=int(output.deletions),
        insertions=int(output.insertions),
        ref_words=len(reference_words),
        hyp_words=len(hyp_words),
    )

    counts = attribute_alignment(
        output.alignments[0], hyp_offsets, list(chunk_durations)
    )

    _assert_attribution_conserved(global_stats, counts)
    return global_stats, counts


def attribute_alignment(
    ops: Sequence[Any],
    hyp_offsets: Sequence[int],
    chunk_durations: Sequence[float],
) -> list[ChunkAlignmentCounts]:
    """Attribute jiwer alignment ops to chunks via hypothesis-word spans."""
    num_chunks = len(hyp_offsets) - 1
    if num_chunks != len(chunk_durations):
        raise ValueError("hyp_offsets and chunk_durations are inconsistent")
    counts = [ChunkAlignmentCounts() for _ in range(num_chunks)]

    for op in ops:
        if op.type == "equal":
            for chunk_idx, amount in _spread_hyp_range(
                op.hyp_start_idx, op.hyp_end_idx, hyp_offsets
            ):
                counts[chunk_idx].hits += amount
                counts[chunk_idx].ref_words += amount
        elif op.type == "substitute":
            for chunk_idx, amount in _spread_hyp_range(
                op.hyp_start_idx, op.hyp_end_idx, hyp_offsets
            ):
                counts[chunk_idx].substitutions += amount
                counts[chunk_idx].ref_words += amount
        elif op.type == "insert":
            for chunk_idx, amount in _spread_hyp_range(
                op.hyp_start_idx, op.hyp_end_idx, hyp_offsets
            ):
                counts[chunk_idx].insertions += amount
        elif op.type == "delete":
            num_deleted = op.ref_end_idx - op.ref_start_idx
            for chunk_idx, weight in _deletion_shares(
                op.hyp_start_idx, hyp_offsets, chunk_durations
            ):
                counts[chunk_idx].deletions += num_deleted * weight
                counts[chunk_idx].ref_words += num_deleted * weight
        else:
            raise ValueError(f"Unknown jiwer alignment op type: {op.type}")

    return counts


def _spread_hyp_range(
    hyp_start: int, hyp_end: int, hyp_offsets: Sequence[int]
) -> list[tuple[int, int]]:
    """Split hypothesis-word range [hyp_start, hyp_end) across chunk spans."""
    num_chunks = len(hyp_offsets) - 1
    result: list[tuple[int, int]] = []
    first = max(0, bisect.bisect_right(hyp_offsets, hyp_start) - 1)
    for chunk_idx in range(first, num_chunks):
        chunk_start = hyp_offsets[chunk_idx]
        chunk_end = hyp_offsets[chunk_idx + 1]
        if chunk_start >= hyp_end:
            break
        overlap = min(hyp_end, chunk_end) - max(hyp_start, chunk_start)
        if overlap > 0:
            result.append((chunk_idx, overlap))
    return result


def _deletion_shares(
    hyp_position: int,
    hyp_offsets: Sequence[int],
    chunk_durations: Sequence[float],
) -> list[tuple[int, float]]:
    """Chunks (with duration-proportional weights) owning a deletion at
    hypothesis position ``hyp_position``."""
    num_chunks = len(hyp_offsets) - 1
    containing = min(
        max(0, bisect.bisect_right(hyp_offsets, hyp_position) - 1), num_chunks - 1
    )
    candidates = {containing}
    lo = bisect.bisect_left(hyp_offsets, hyp_position)
    hi = bisect.bisect_right(hyp_offsets, hyp_position)
    for offset_idx in range(lo, hi):
        chunk_idx = offset_idx
        if (
            chunk_idx < num_chunks
            and hyp_offsets[chunk_idx] == hyp_offsets[chunk_idx + 1] == hyp_position
        ):
            candidates.add(chunk_idx)

    ordered = sorted(candidates)
    total_duration = sum(chunk_durations[idx] for idx in ordered)
    if total_duration <= 0.0:
        uniform = 1.0 / len(ordered)
        return [(idx, uniform) for idx in ordered]
    return [(idx, chunk_durations[idx] / total_duration) for idx in ordered]


def _assert_attribution_conserved(
    global_stats: GlobalWerStats, counts: Sequence[ChunkAlignmentCounts]
) -> None:
    totals = {
        "hits": sum(c.hits for c in counts),
        "substitutions": sum(c.substitutions for c in counts),
        "deletions": sum(c.deletions for c in counts),
        "insertions": sum(c.insertions for c in counts),
        "ref_words": sum(c.ref_words for c in counts),
    }
    expected = {
        "hits": global_stats.hits,
        "substitutions": global_stats.substitutions,
        "deletions": global_stats.deletions,
        "insertions": global_stats.insertions,
        "ref_words": global_stats.ref_words,
    }
    for key, total in totals.items():
        if not math.isclose(total, expected[key], rel_tol=0.0, abs_tol=1e-6):
            raise AssertionError(
                f"Per-chunk attribution lost {key}: {total} != {expected[key]}"
            )


# ---------------------------------------------------------------------------
# Bucket assembly
# ---------------------------------------------------------------------------


def bucket_weighted(
    spans: Sequence[tuple[float, float]],
    values: Sequence[dict[str, float]],
    bucket_seconds: float,
    num_buckets: int,
) -> list[dict[str, float]]:
    """Accumulate per-span additive values into fixed time buckets, weighting
    each span's contribution by its time overlap with the bucket."""
    if len(spans) != len(values):
        raise ValueError("spans and values length mismatch")
    buckets: list[dict[str, float]] = [dict() for _ in range(num_buckets)]
    for (start_s, end_s), value_map in zip(spans, values):
        duration = end_s - start_s
        if duration <= 0:
            raise ValueError(f"Empty span [{start_s}, {end_s})")
        first_bucket = int(start_s // bucket_seconds)
        last_bucket = min(int(math.ceil(end_s / bucket_seconds)) - 1, num_buckets - 1)
        for bucket_idx in range(first_bucket, last_bucket + 1):
            bucket_start = bucket_idx * bucket_seconds
            bucket_end = bucket_start + bucket_seconds
            overlap = min(end_s, bucket_end) - max(start_s, bucket_start)
            if overlap <= 0:
                continue
            weight = overlap / duration
            target = buckets[bucket_idx]
            for key, value in value_map.items():
                target[key] = target.get(key, 0.0) + weight * value
    return buckets


# ---------------------------------------------------------------------------
# Energy track
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyBin:
    index: int
    start_s: float
    end_s: float
    mean_square: float
    rms_dbfs: float
    silence_frac: float


def rms_to_dbfs(mean_square: float) -> float:
    if mean_square <= 0.0:
        return _DBFS_FLOOR
    return max(10.0 * math.log10(mean_square), _DBFS_FLOOR)


def compute_energy_bins(
    audio: AudioSource,
    bin_seconds: float,
    silence_threshold_dbfs: float,
) -> list[EnergyBin]:
    """RMS and silence fraction per fixed bin (30 ms silence frames)."""
    silence_threshold_linear = 10.0 ** (silence_threshold_dbfs / 20.0)
    frame_len = max(1, int(round(_ENERGY_FRAME_SECONDS * audio.sample_rate)))
    bins: list[EnergyBin] = []
    for span in make_chunk_spans(audio.num_samples, audio.sample_rate, bin_seconds):
        chunk = audio.slice_f32(span.start_sample, span.end_sample)
        mean_square = float(np.mean(np.square(chunk)))
        num_frames = len(chunk) // frame_len
        if num_frames > 0:
            frames = chunk[: num_frames * frame_len].reshape(num_frames, frame_len)
            frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
            silence_frac = float(np.mean(frame_rms < silence_threshold_linear))
        else:
            silence_frac = float(math.sqrt(mean_square) < silence_threshold_linear)
        bins.append(
            EnergyBin(
                index=span.index,
                start_s=span.start_s,
                end_s=span.end_s,
                mean_square=mean_square,
                rms_dbfs=rms_to_dbfs(mean_square),
                silence_frac=silence_frac,
            )
        )
    return bins


# ---------------------------------------------------------------------------
# Track results
# ---------------------------------------------------------------------------


@dataclass
class WerChunkResult:
    index: int
    start_s: float
    end_s: float
    transcript: str
    num_hyp_words: int
    ref_words: float
    hits: float
    substitutions: float
    deletions: float
    insertions: float
    wer: Optional[float]
    dup_ngrams: int
    compression_ratio: float
    compression_flagged: bool


@dataclass
class UtmosChunkResult:
    index: int
    start_s: float
    end_s: float
    score: Optional[float]


@dataclass
class SimWindowResult:
    index: int
    start_s: float
    end_s: float
    cosine: float


@dataclass
class BucketRow:
    """One drift-curve bucket (per-minute by default) — the curves.csv row."""

    bucket: int
    start_s: float
    wer: Optional[float] = None
    utmos_mean: Optional[float] = None
    utmos_min: Optional[float] = None
    insertions: Optional[float] = None
    deletions: Optional[float] = None
    dup_ngrams: Optional[float] = None
    rms_dbfs: Optional[float] = None
    silence_frac: Optional[float] = None
    sim_mean: Optional[float] = None
    sim_min: Optional[float] = None


# ---------------------------------------------------------------------------
# Track computations
# ---------------------------------------------------------------------------


def compute_wer_track(
    audio: AudioSource,
    reference_text: str,
    config: ScoreTtsLongformConfig,
    transcriber: ChunkTranscriber,
) -> tuple[GlobalWerStats, list[WerChunkResult], str]:
    """Transcribe 28 s chunks, align once globally, attribute per chunk."""
    reference_words = normalize_text(reference_text).split()
    spans = make_chunk_spans(
        audio.num_samples, audio.sample_rate, config.asr.chunk_seconds
    )
    transcripts: list[str] = []
    chunk_words: list[list[str]] = []
    started = time.monotonic()
    for span in spans:
        transcript = transcriber.transcribe(
            audio.slice_f32_16k(span.start_sample, span.end_sample)
        )
        transcripts.append(transcript)
        chunk_words.append(normalize_text(transcript).split())
        if (span.index + 1) % _PROGRESS_LOG_EVERY == 0:
            elapsed = time.monotonic() - started
            logger.info(
                "WER track: transcribed %d/%d chunks (%.1fs elapsed)",
                span.index + 1,
                len(spans),
                elapsed,
            )

    global_stats, chunk_counts = align_chunks_to_reference(
        reference_words,
        chunk_words,
        [span.duration_s for span in spans],
    )

    chunk_results: list[WerChunkResult] = []
    for span, transcript, words, counts in zip(
        spans, transcripts, chunk_words, chunk_counts
    ):
        ratio = compression_ratio(transcript)
        chunk_results.append(
            WerChunkResult(
                index=span.index,
                start_s=span.start_s,
                end_s=span.end_s,
                transcript=transcript,
                num_hyp_words=len(words),
                ref_words=counts.ref_words,
                hits=counts.hits,
                substitutions=counts.substitutions,
                deletions=counts.deletions,
                insertions=counts.insertions,
                wer=counts.wer,
                dup_ngrams=count_duplicated_ngrams(words, config.dup_ngram_size),
                compression_ratio=ratio,
                compression_flagged=ratio > config.compression_ratio_threshold,
            )
        )
    concatenated_transcript = " ".join(t for t in transcripts if t)
    return global_stats, chunk_results, concatenated_transcript


def compute_utmos_track(
    audio: AudioSource,
    config: LongformUtmosConfig,
    scorer: UtmosScorer,
) -> list[UtmosChunkResult]:
    """Score 10 s chunks with UTMOS at 16 kHz."""
    spans = make_chunk_spans(
        audio.num_samples,
        audio.sample_rate,
        config.chunk_seconds,
        min_tail_seconds=config.min_chunk_seconds,
    )
    results: list[UtmosChunkResult] = []
    for span in spans:
        score = scorer.score(audio.slice_f32_16k(span.start_sample, span.end_sample))
        results.append(
            UtmosChunkResult(
                index=span.index,
                start_s=span.start_s,
                end_s=span.end_s,
                score=score,
            )
        )
    return results


def compute_sim_track(
    audio: AudioSource,
    config: LongformSimConfig,
    embedder: SimEmbedder,
) -> tuple[list[SimWindowResult], str]:
    """Cosine similarity of 3 s window embeddings vs the anchor embedding."""
    if config.prompt_audio:
        prompt_path = Path(config.prompt_audio)
        if not prompt_path.exists():
            raise FileNotFoundError(f"SIM prompt audio not found: {prompt_path}")
        prompt = AudioSource(prompt_path, MODEL_SAMPLE_RATE)
        anchor_wav = prompt.slice_f32_16k(0, prompt.num_samples)
        anchor_source = f"prompt_audio:{prompt_path}"
    else:
        anchor_samples = min(
            int(round(config.reference_seconds * audio.sample_rate)),
            audio.num_samples,
        )
        anchor_wav = audio.slice_f32_16k(0, anchor_samples)
        anchor_source = f"audio_head:{config.reference_seconds}s"

    anchor = embedder.embed(anchor_wav)
    anchor_norm = float(np.linalg.norm(anchor))
    if anchor_norm <= 0.0:
        raise ValueError("SIM anchor embedding has zero norm")

    spans = make_chunk_spans(
        audio.num_samples,
        audio.sample_rate,
        config.window_seconds,
        min_tail_seconds=config.window_seconds,
    )
    results: list[SimWindowResult] = []
    for span in spans:
        embedding = embedder.embed(
            audio.slice_f32_16k(span.start_sample, span.end_sample)
        )
        norm = float(np.linalg.norm(embedding))
        if norm <= 0.0:
            continue
        cosine = float(np.dot(anchor, embedding) / (anchor_norm * norm))
        results.append(
            SimWindowResult(
                index=span.index,
                start_s=span.start_s,
                end_s=span.end_s,
                cosine=cosine,
            )
        )
    return results, anchor_source


# ---------------------------------------------------------------------------
# Bucket curve assembly
# ---------------------------------------------------------------------------


def assemble_bucket_rows(
    duration_s: float,
    bucket_seconds: float,
    wer_chunks: Optional[list[WerChunkResult]],
    utmos_chunks: Optional[list[UtmosChunkResult]],
    sim_windows: Optional[list[SimWindowResult]],
    energy_bins: list[EnergyBin],
) -> list[BucketRow]:
    num_buckets = max(1, int(math.ceil(duration_s / bucket_seconds)))
    rows = [
        BucketRow(bucket=idx, start_s=idx * bucket_seconds)
        for idx in range(num_buckets)
    ]

    if wer_chunks:
        wer_buckets = bucket_weighted(
            [(c.start_s, c.end_s) for c in wer_chunks],
            [
                {
                    "ref_words": c.ref_words,
                    "errors": c.substitutions + c.deletions + c.insertions,
                    "insertions": c.insertions,
                    "deletions": c.deletions,
                    "dup_ngrams": float(c.dup_ngrams),
                }
                for c in wer_chunks
            ],
            bucket_seconds,
            num_buckets,
        )
        for row, accumulated in zip(rows, wer_buckets):
            if not accumulated:
                continue
            ref_words = accumulated.get("ref_words", 0.0)
            if ref_words > 0.0:
                row.wer = accumulated.get("errors", 0.0) / ref_words
            row.insertions = accumulated.get("insertions", 0.0)
            row.deletions = accumulated.get("deletions", 0.0)
            row.dup_ngrams = accumulated.get("dup_ngrams", 0.0)

    if utmos_chunks:
        scored: dict[int, list[float]] = {}
        for chunk in utmos_chunks:
            if chunk.score is None:
                continue
            bucket_idx = min(int(chunk.start_s // bucket_seconds), num_buckets - 1)
            scored.setdefault(bucket_idx, []).append(chunk.score)
        for bucket_idx, values in scored.items():
            rows[bucket_idx].utmos_mean = float(np.mean(values))
            rows[bucket_idx].utmos_min = float(np.min(values))

    if sim_windows:
        sim_scored: dict[int, list[float]] = {}
        for window in sim_windows:
            bucket_idx = min(int(window.start_s // bucket_seconds), num_buckets - 1)
            sim_scored.setdefault(bucket_idx, []).append(window.cosine)
        for bucket_idx, values in sim_scored.items():
            rows[bucket_idx].sim_mean = float(np.mean(values))
            rows[bucket_idx].sim_min = float(np.min(values))

    energy_buckets = bucket_weighted(
        [(b.start_s, b.end_s) for b in energy_bins],
        [
            {
                "mean_square_x_dur": b.mean_square * (b.end_s - b.start_s),
                "silence_x_dur": b.silence_frac * (b.end_s - b.start_s),
                "dur": b.end_s - b.start_s,
            }
            for b in energy_bins
        ],
        bucket_seconds,
        num_buckets,
    )
    for row, accumulated in zip(rows, energy_buckets):
        duration = accumulated.get("dur", 0.0)
        if duration <= 0.0:
            continue
        row.rms_dbfs = rms_to_dbfs(accumulated["mean_square_x_dur"] / duration)
        row.silence_frac = accumulated["silence_x_dur"] / duration

    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

_CURVES_COLUMNS = [
    "bucket",
    "wer",
    "utmos_mean",
    "utmos_min",
    "ins",
    "del",
    "dup5grams",
    "rms",
    "silence_frac",
]


def _format_cell(value: Optional[float], digits: int) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_curves_csv(path: Path, rows: list[BucketRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CURVES_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row.bucket,
                    _format_cell(row.wer, 4),
                    _format_cell(row.utmos_mean, 3),
                    _format_cell(row.utmos_min, 3),
                    _format_cell(row.insertions, 2),
                    _format_cell(row.deletions, 2),
                    _format_cell(row.dup_ngrams, 2),
                    _format_cell(row.rms_dbfs, 2),
                    _format_cell(row.silence_frac, 4),
                ]
            )


def _mean_of(values: list[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _drift_line(
    label: str, rows: list[BucketRow], attr: str, fmt: str
) -> Optional[str]:
    values = [getattr(row, attr) for row in rows]
    present = [(idx, v) for idx, v in enumerate(values) if v is not None]
    if len(present) < 4:
        return None
    head = [v for _, v in present[:2]]
    tail = [v for _, v in present[-2:]]
    head_mean = float(np.mean(head))
    tail_mean = float(np.mean(tail))
    return (
        f"  {label}: first-2 {head_mean:{fmt}} -> last-2 {tail_mean:{fmt}} "
        f"(delta {tail_mean - head_mean:+{fmt}})"
    )


def write_report_txt(
    path: Path,
    summary: dict[str, Any],
    rows: list[BucketRow],
) -> None:
    lines: list[str] = []
    audio_info = summary["audio"]
    lines.append("TTS long-form position-resolved quality report")
    lines.append("=" * 47)
    lines.append(
        f"audio: {audio_info['path']} "
        f"({audio_info['duration_s']:.1f}s = {audio_info['duration_s'] / 60:.1f}min, "
        f"{audio_info['sample_rate']} Hz)"
    )
    reference = summary["reference"]
    lines.append(
        f"reference: {reference['path']} ({reference['num_words']} normalized words)"
    )
    for note in summary["notes"]:
        lines.append(f"note: {note}")
    lines.append("")

    lines.append("GLOBAL")
    wer_summary = summary["wer"]
    if wer_summary is not None:
        global_wer = wer_summary["global"]
        lines.append(
            f"  WER: {100.0 * global_wer['wer']:.2f}%  "
            f"(S={global_wer['substitutions']} D={global_wer['deletions']} "
            f"I={global_wer['insertions']} H={global_wer['hits']} "
            f"ref={global_wer['ref_words']} hyp={global_wer['hyp_words']})"
        )
        repetition = summary["repetition"]
        lines.append(
            f"  loops: duplicated {repetition['ngram_size']}-grams="
            f"{repetition['total_dup_ngrams']}, compression-flagged chunks="
            f"{repetition['flagged_chunks']}/{repetition['total_chunks']} "
            f"(zlib ratio > {repetition['compression_ratio_threshold']})"
        )
    else:
        lines.append("  WER: skipped")
    utmos_summary = summary["utmos"]
    if utmos_summary is not None:
        lines.append(
            f"  UTMOS: mean {utmos_summary['global_mean']:.3f}  "
            f"min {utmos_summary['global_min']:.3f}  "
            f"({utmos_summary['scored_chunks']} chunks)"
        )
    else:
        lines.append("  UTMOS: skipped")
    sim_summary = summary["sim"]
    if sim_summary is not None:
        lines.append(
            f"  SIM: mean {sim_summary['global_mean']:.4f}  "
            f"min {sim_summary['global_min']:.4f}  "
            f"({sim_summary['num_windows']} windows, anchor "
            f"{sim_summary['anchor_source']})"
        )
    else:
        lines.append("  SIM: skipped")
    energy = summary["energy"]
    lines.append(
        f"  energy: RMS {energy['global_rms_dbfs']:.2f} dBFS, "
        f"silence {100.0 * energy['global_silence_frac']:.1f}%"
    )
    lines.append("")

    drift_lines = [
        _drift_line("WER", rows, "wer", ".4f"),
        _drift_line("UTMOS mean", rows, "utmos_mean", ".3f"),
        _drift_line("SIM mean", rows, "sim_mean", ".4f"),
        _drift_line("silence frac", rows, "silence_frac", ".4f"),
    ]
    drift_lines = [line for line in drift_lines if line is not None]
    if drift_lines:
        lines.append("DRIFT (mean of first 2 vs last 2 populated buckets)")
        lines.extend(drift_lines)
        lines.append("")

    lines.append("PER-BUCKET CURVES (bucket size %.0fs)" % summary["bucket_seconds"])
    header = (
        f"{'bucket':>6} {'wer%':>8} {'utmos':>7} {'utmos_min':>9} "
        f"{'ins':>7} {'del':>7} {'dup':>6} {'rms_dbfs':>9} {'silence%':>9}"
    )
    lines.append(header)
    for row in rows:
        wer_cell = f"{100.0 * row.wer:.2f}" if row.wer is not None else "-"
        lines.append(
            f"{row.bucket:>6} "
            f"{wer_cell:>8} "
            f"{_format_cell(row.utmos_mean, 2) or '-':>7} "
            f"{_format_cell(row.utmos_min, 2) or '-':>9} "
            f"{_format_cell(row.insertions, 1) or '-':>7} "
            f"{_format_cell(row.deletions, 1) or '-':>7} "
            f"{_format_cell(row.dup_ngrams, 1) or '-':>6} "
            f"{_format_cell(row.rms_dbfs, 2) or '-':>9} "
            f"{(f'{100.0 * row.silence_frac:.1f}' if row.silence_frac is not None else '-'):>9}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class LongformScoreResult:
    summary: dict[str, Any]
    bucket_rows: list[BucketRow] = field(repr=False, default_factory=list)
    summary_path: Optional[Path] = None
    curves_path: Optional[Path] = None
    report_path: Optional[Path] = None


def run_score_tts_longform(
    config: ScoreTtsLongformConfig,
    transcriber: Optional[ChunkTranscriber] = None,
    utmos_scorer: Optional[UtmosScorer] = None,
    sim_embedder: Optional[SimEmbedder] = None,
) -> LongformScoreResult:
    """Score one long-form waveform and write summary/curves/report files.

    Backends may be injected (tests); by default they are built from the
    config and each track degrades to a skip-note when its model is
    unavailable. The model-free energy track always runs.
    """
    audio_path = Path(config.audio)
    reference_path = Path(config.reference_text)
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference text file not found: {reference_path}")
    reference_text = " ".join(
        line.strip()
        for line in reference_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not reference_text:
        raise ValueError(f"Reference text file is empty: {reference_path}")

    audio = AudioSource(audio_path, config.sample_rate)
    logger.info(
        "Scoring %s: %.1fs (%.1f min) at %d Hz",
        audio_path,
        audio.duration_s,
        audio.duration_s / 60,
        audio.sample_rate,
    )

    notes: list[str] = []
    if transcriber is None:
        transcriber, note = build_transcriber(config.asr)
        if note:
            notes.append(note)
    if utmos_scorer is None:
        utmos_scorer, note = build_utmos_scorer(config.utmos)
        if note:
            notes.append(note)
    if sim_embedder is None:
        sim_embedder, note = build_sim_embedder(config.sim)
        if note:
            notes.append(note)
    for note in notes:
        logger.warning("%s", note)

    global_wer: Optional[GlobalWerStats] = None
    wer_chunks: Optional[list[WerChunkResult]] = None
    concatenated_transcript = ""
    if transcriber is not None:
        global_wer, wer_chunks, concatenated_transcript = compute_wer_track(
            audio, reference_text, config, transcriber
        )

    utmos_chunks: Optional[list[UtmosChunkResult]] = None
    if utmos_scorer is not None:
        utmos_chunks = compute_utmos_track(audio, config.utmos, utmos_scorer)

    sim_windows: Optional[list[SimWindowResult]] = None
    sim_anchor_source = ""
    if sim_embedder is not None:
        sim_windows, sim_anchor_source = compute_sim_track(
            audio, config.sim, sim_embedder
        )

    energy_bins = compute_energy_bins(
        audio, config.energy_bin_seconds, config.silence_threshold_dbfs
    )

    bucket_rows = assemble_bucket_rows(
        audio.duration_s,
        config.bucket_seconds,
        wer_chunks,
        utmos_chunks,
        sim_windows,
        energy_bins,
    )

    summary = _build_summary(
        config,
        audio,
        reference_text,
        notes,
        global_wer,
        wer_chunks,
        concatenated_transcript,
        utmos_chunks,
        sim_windows,
        sim_anchor_source,
        energy_bins,
        bucket_rows,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    curves_path = output_dir / "curves.csv"
    report_path = output_dir / "report.txt"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_curves_csv(curves_path, bucket_rows)
    write_report_txt(report_path, summary, bucket_rows)
    logger.info("Wrote %s, %s, %s", summary_path, curves_path, report_path)

    return LongformScoreResult(
        summary=summary,
        bucket_rows=bucket_rows,
        summary_path=summary_path,
        curves_path=curves_path,
        report_path=report_path,
    )


def run_score_tts_longform_cli(configs: list[ScoreTtsLongformConfig]) -> None:
    for config in configs:
        run_score_tts_longform(config)


def _build_summary(
    config: ScoreTtsLongformConfig,
    audio: AudioSource,
    reference_text: str,
    notes: list[str],
    global_wer: Optional[GlobalWerStats],
    wer_chunks: Optional[list[WerChunkResult]],
    concatenated_transcript: str,
    utmos_chunks: Optional[list[UtmosChunkResult]],
    sim_windows: Optional[list[SimWindowResult]],
    sim_anchor_source: str,
    energy_bins: list[EnergyBin],
    bucket_rows: list[BucketRow],
) -> dict[str, Any]:
    reference_words = normalize_text(reference_text).split()

    wer_summary: Optional[dict[str, Any]] = None
    repetition_summary: Optional[dict[str, Any]] = None
    if global_wer is not None and wer_chunks is not None:
        wer_summary = {
            "global": asdict(global_wer),
            "chunk_seconds": config.asr.chunk_seconds,
            "asr_model": config.asr.model,
            "asr_backend": "faster-whisper (greedy, language pinned, VAD off; "
            "seed protocol uses HF whisper-large-v3 greedy)",
            "alignment_method": "single global jiwer alignment; per-chunk "
            "attribution by hypothesis-word spans, deletions split "
            "duration-proportionally with empty chunks at the same position",
            "chunks": [asdict(chunk) for chunk in wer_chunks],
            "concatenated_transcript": concatenated_transcript,
        }
        repetition_summary = {
            "ngram_size": config.dup_ngram_size,
            "total_dup_ngrams": sum(chunk.dup_ngrams for chunk in wer_chunks),
            "flagged_chunks": sum(
                1 for chunk in wer_chunks if chunk.compression_flagged
            ),
            "total_chunks": len(wer_chunks),
            "compression_ratio_threshold": config.compression_ratio_threshold,
            "total_insertions": global_wer.insertions,
            "total_deletions": global_wer.deletions,
        }

    utmos_summary: Optional[dict[str, Any]] = None
    if utmos_chunks is not None:
        scores = [chunk.score for chunk in utmos_chunks if chunk.score is not None]
        if scores:
            utmos_summary = {
                "global_mean": float(np.mean(scores)),
                "global_min": float(np.min(scores)),
                "scored_chunks": len(scores),
                "total_chunks": len(utmos_chunks),
                "chunk_seconds": config.utmos.chunk_seconds,
                "chunks": [asdict(chunk) for chunk in utmos_chunks],
            }
        else:
            notes = notes + ["UTMOS produced no finite scores"]

    sim_summary: Optional[dict[str, Any]] = None
    if sim_windows is not None and sim_windows:
        cosines = [window.cosine for window in sim_windows]
        sim_summary = {
            "global_mean": float(np.mean(cosines)),
            "global_min": float(np.min(cosines)),
            "num_windows": len(sim_windows),
            "window_seconds": config.sim.window_seconds,
            "anchor_source": sim_anchor_source,
            "windows": [asdict(window) for window in sim_windows],
        }

    total_duration = sum(b.end_s - b.start_s for b in energy_bins)
    global_mean_square = (
        sum(b.mean_square * (b.end_s - b.start_s) for b in energy_bins) / total_duration
    )
    global_silence = (
        sum(b.silence_frac * (b.end_s - b.start_s) for b in energy_bins)
        / total_duration
    )
    energy_summary = {
        "bin_seconds": config.energy_bin_seconds,
        "silence_threshold_dbfs": config.silence_threshold_dbfs,
        "global_rms_dbfs": rms_to_dbfs(global_mean_square),
        "global_silence_frac": global_silence,
        "bins": [asdict(b) for b in energy_bins],
    }

    return {
        "audio": {
            "path": str(audio.path),
            "sample_rate": audio.sample_rate,
            "num_samples": audio.num_samples,
            "duration_s": audio.duration_s,
        },
        "reference": {
            "path": config.reference_text,
            "num_words": len(reference_words),
        },
        "bucket_seconds": config.bucket_seconds,
        "notes": notes,
        "wer": wer_summary,
        "repetition": repetition_summary,
        "utmos": utmos_summary,
        "sim": sim_summary,
        "energy": energy_summary,
        "buckets": [asdict(row) for row in bucket_rows],
    }
