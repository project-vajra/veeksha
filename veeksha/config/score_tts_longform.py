"""Configuration for the `veeksha score-tts-longform` command.

Position-resolved quality scoring for a single long-form TTS waveform,
following the windowed-KV quality-ladder methodology (Seed-TTS-Eval exact
normalization, 28 s WER chunks, 10 s UTMOS chunks, 3 s SIM windows,
per-minute drift buckets).
"""

from vidhi import field, frozen_dataclass

from veeksha.cli.base import VeekshaCommand

_SIM_CHECKPOINT_HELP = (
    "TorchScript export of the WavLM-large speaker-verification model "
    "(Microsoft UniSpeech `wavlm_large_finetune.pth`, WavLM-large + "
    "ECAPA-TDNN head). Download source: "
    "https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification "
    "(the README links the checkpoint). The raw state-dict checkpoint must be "
    "exported to TorchScript with the UniSpeech model code before use here; a "
    "raw state dict cannot be loaded standalone. Empty path skips SIM scoring "
    "with a note."
)


@frozen_dataclass
class LongformAsrConfig:
    """ASR settings for the position-resolved WER track.

    Seed-TTS-Eval scores English WER with HF `openai/whisper-large-v3` and
    GREEDY decoding. This command uses faster-whisper (CTranslate2) with
    beam_size=1 / temperature=0.0, language pinned, VAD off, and no
    conditioning on previous text — the closest faster-whisper equivalent of
    the seed protocol. Backend difference (CTranslate2 vs HF transformers) is
    the one documented deviation.
    """

    model: str = field(
        "large-v3",
        help="Whisper model identifier passed to faster-whisper.",
    )
    device: str = field(
        "auto",
        help="Device passed to faster-whisper: 'auto', 'cuda', or 'cpu'.",
    )
    compute_type: str = field(
        "default",
        help="Compute type passed to faster-whisper, e.g. 'float16' or 'int8'.",
    )
    language: str = field(
        "en",
        help="Language pinned for transcription (seed protocol pins English).",
    )
    chunk_seconds: float = field(
        28.0,
        help=(
            "Non-overlapping transcription chunk length in seconds. 28 s keeps "
            "every chunk under Whisper's 30 s window (Long-TTS-Eval "
            "convention); never feed >30 s to Whisper."
        ),
    )

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("LongformAsrConfig.model is required")
        if not self.device:
            raise ValueError("LongformAsrConfig.device is required")
        if not self.compute_type:
            raise ValueError("LongformAsrConfig.compute_type is required")
        if not self.language:
            raise ValueError("LongformAsrConfig.language is required")
        if not 0 < self.chunk_seconds <= 30.0:
            raise ValueError(
                "LongformAsrConfig.chunk_seconds must be in (0, 30] to stay "
                "inside Whisper's 30 s window"
            )


@frozen_dataclass
class LongformUtmosConfig:
    """UTMOS (balacoon TorchScript) settings for the naturalness track."""

    hf_repo: str = field(
        "balacoon/utmos",
        help="Hugging Face model repo containing the UTMOS JIT file.",
    )
    jit_file: str = field(
        "utmos.jit",
        help="TorchScript filename to load from the UTMOS HF repo.",
    )
    device: str = field(
        "cpu",
        help="Device for UTMOS TorchScript inference, e.g. 'cuda:0' or 'cpu'.",
    )
    chunk_seconds: float = field(
        10.0,
        help=(
            "Non-overlapping UTMOS chunk length in seconds (field practice for "
            "long audio: chunk-then-aggregate; report mean AND min)."
        ),
    )
    min_chunk_seconds: float = field(
        1.0,
        help="Trailing chunks shorter than this are dropped (unstable scores).",
    )

    def __post_init__(self) -> None:
        if not self.hf_repo:
            raise ValueError("LongformUtmosConfig.hf_repo is required")
        if not self.jit_file:
            raise ValueError("LongformUtmosConfig.jit_file is required")
        if not self.device:
            raise ValueError("LongformUtmosConfig.device is required")
        if self.chunk_seconds <= 0:
            raise ValueError("LongformUtmosConfig.chunk_seconds must be positive")
        if self.min_chunk_seconds < 0:
            raise ValueError(
                "LongformUtmosConfig.min_chunk_seconds must be non-negative"
            )


@frozen_dataclass
class LongformSimConfig:
    """Speaker-similarity (WavLM-SV) drift-curve settings.

    SIM(t) is the leading indicator of long-form drift (MOSS-TTS): cosine
    similarity between each non-overlapping 3 s window embedding and a
    reference embedding, bucketed over time.
    """

    checkpoint_path: str = field("", help=_SIM_CHECKPOINT_HELP)
    prompt_audio: str = field(
        "",
        help=(
            "Optional prompt/reference wav for the SIM anchor embedding. If "
            "empty, the first `reference_seconds` of the scored audio anchor "
            "the curve (self-drift mode)."
        ),
    )
    device: str = field(
        "cpu",
        help="Device for WavLM-SV inference, e.g. 'cuda:0' or 'cpu'.",
    )
    window_seconds: float = field(
        3.0,
        help="Non-overlapping SIM window length in seconds (MOSS-TTS recipe).",
    )
    reference_seconds: float = field(
        10.0,
        help="Anchor length taken from the audio head in self-drift mode.",
    )

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("LongformSimConfig.device is required")
        if self.window_seconds <= 0:
            raise ValueError("LongformSimConfig.window_seconds must be positive")
        if self.reference_seconds <= 0:
            raise ValueError("LongformSimConfig.reference_seconds must be positive")


@frozen_dataclass
class ScoreTtsLongformConfig(VeekshaCommand, name="score-tts-longform"):
    """Score a long-form TTS waveform with position-resolved quality metrics.

    Produces summary.json, curves.csv (per-minute buckets), and report.txt in
    the output directory. Tracks: WER(t) via chunked Whisper + one global
    jiwer alignment, UTMOS(t), repetition/omission detectors, RMS/silence
    energy, and optional WavLM-SV speaker-similarity drift.
    """

    audio: str = field(
        "",
        help=(
            "Path to the audio to score: .wav (any sample rate, converted to "
            "mono), or raw PCM (int16 little-endian mono) for any other "
            "extension."
        ),
    )
    sample_rate: int = field(
        24000,
        aliases=["sample-rate"],
        help="Sample rate of raw PCM input. Ignored for WAV (header wins).",
    )
    reference_text: str = field(
        "",
        aliases=["reference-text"],
        help=(
            "Path to a UTF-8 text file with the reference text; lines are "
            "joined with single spaces before alignment."
        ),
    )
    output_dir: str = field(
        "longform_scores",
        aliases=["output-dir"],
        help="Directory for summary.json, curves.csv, and report.txt.",
    )
    bucket_seconds: float = field(
        60.0,
        help="Drift-curve bucket size in seconds (per-minute by default).",
    )
    energy_bin_seconds: float = field(
        30.0,
        help="Bin size for the RMS/silence energy track.",
    )
    silence_threshold_dbfs: float = field(
        -40.0,
        help="Frame RMS below this dBFS level counts as silence.",
    )
    dup_ngram_size: int = field(
        5,
        help="N-gram size for the duplicated-n-gram loop detector (WhisperX uses 5).",
    )
    compression_ratio_threshold: float = field(
        2.4,
        help=(
            "Per-chunk transcript zlib compression ratio above this flags a "
            "likely loop (Whisper convention)."
        ),
    )
    asr: LongformAsrConfig = field(
        default_factory=LongformAsrConfig,
        help="ASR settings for the WER track.",
    )
    utmos: LongformUtmosConfig = field(
        default_factory=LongformUtmosConfig,
        help="UTMOS settings for the naturalness track.",
    )
    sim: LongformSimConfig = field(
        default_factory=LongformSimConfig,
        help="Speaker-similarity (WavLM-SV) drift settings.",
    )

    def __post_init__(self) -> None:
        if not self.audio:
            raise ValueError("score-tts-longform requires --audio")
        if not self.reference_text:
            raise ValueError("score-tts-longform requires --reference_text")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.bucket_seconds <= 0:
            raise ValueError("bucket_seconds must be positive")
        if self.energy_bin_seconds <= 0:
            raise ValueError("energy_bin_seconds must be positive")
        if self.dup_ngram_size < 2:
            raise ValueError("dup_ngram_size must be >= 2")
        if self.compression_ratio_threshold <= 0:
            raise ValueError("compression_ratio_threshold must be positive")
