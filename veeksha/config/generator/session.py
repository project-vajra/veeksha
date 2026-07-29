from vidhi import BasePolyConfig, field, frozen_dataclass

from veeksha.config.generator.channel import (
    BaseChannelGeneratorConfig,
    TextChannelGeneratorConfig,
)
from veeksha.config.generator.requested_output import OutputSpecConfig
from veeksha.config.generator.session_graph import (
    BaseSessionGraphGeneratorConfig,
    LinearSessionGraphGeneratorConfig,
)
from veeksha.types import (
    SessionGeneratorType,
    TraceFlavorType,
)


@frozen_dataclass
class BaseSessionGeneratorConfig(BasePolyConfig):
    """Session generator strategy (synthetic, trace, or lm-eval)."""


@frozen_dataclass
class SyntheticSessionGeneratorConfig(BaseSessionGeneratorConfig):
    """Configuration for synthetic session generation.

    Attributes:
        session_graph: Configuration for session graph structure.
        channels: Input channel configurations (text, image, etc.).
        output_spec: Specification for expected output from the model.
    """

    session_graph: BaseSessionGraphGeneratorConfig = field(
        default_factory=LinearSessionGraphGeneratorConfig,
        help="The generator for the session graphs.",
    )
    channels: list[BaseChannelGeneratorConfig] = field(
        default_factory=lambda: [TextChannelGeneratorConfig()],
        help="The modality channels for the input content of each request.",
    )
    output_spec: OutputSpecConfig = field(
        default_factory=OutputSpecConfig,
        help="Specification for expected output from the model, for supported modalities (e.g., output token length, image count).",
    )

    @classmethod
    def get_type(cls):
        return SessionGeneratorType.SYNTHETIC

    def __post_init__(self):
        channel_types = set([channel.get_type() for channel in self.channels])
        if len(channel_types) != len(self.channels):
            raise ValueError("All channel generators must have unique types")

        if not self.channels:
            raise ValueError("At least one channel generator must be specified")


@frozen_dataclass
class LmevalSessionGeneratorConfig(BaseSessionGeneratorConfig):
    tasks: list[str] = field(
        default_factory=lambda: ["hellaswag"],
        help="The lm-eval tasks to evaluate the model on.",
    )
    num_fewshot: int = field(
        1, help="The number of fewshot examples to use for the tasks."
    )
    # NOTE: We intentionally do not expose a separate `limit` knob here.
    # Control total evaluated sessions via `runtime.max_sessions` (and wall time via
    # `runtime.benchmark_timeout`) to keep run termination consistent across workloads.

    @classmethod
    def get_type(cls):
        return SessionGeneratorType.LMEVAL

    def __post_init__(self):
        if not self.tasks:
            raise ValueError("LmevalSessionGeneratorConfig requires at least one task.")


# ----- Trace Flavor Configs -----


@frozen_dataclass
class BaseTraceFlavorConfig(BasePolyConfig):
    """Base config for trace flavors."""


@frozen_dataclass
class TimedSyntheticSessionTraceFlavorConfig(BaseTraceFlavorConfig):
    """Timed synthetic session trace flavor configuration with context caching."""

    corpus_file: str = field(
        "traces/corpus.txt", help="Path to corpus file for prompt padding"
    )
    page_size: int = field(
        16, help="Number of unique tokens per history-lineage prefix"
    )

    @classmethod
    def get_type(cls):
        return TraceFlavorType.TIMED_SYNTHETIC_SESSION


@frozen_dataclass
class SharedPrefixTraceFlavorConfig(BaseTraceFlavorConfig):
    """Shared-prefix trace flavor configuration with hash-based content sharing."""

    corpus_file: str = field(
        "traces/corpus.txt", help="Path to corpus file for prompt padding"
    )
    block_size: int = field(
        512,
        help="Number of tokens per hash id block. Only used for hash ids of first-in-session requests.",
    )

    @classmethod
    def get_type(cls):
        return TraceFlavorType.SHARED_PREFIX


@frozen_dataclass
class RAGTraceFlavorConfig(BaseTraceFlavorConfig):
    """RAG trace flavor configuration."""

    num_documents: int = field(10, help="Number of top documents to include for warmup")

    @classmethod
    def get_type(cls):
        return TraceFlavorType.RAG


@frozen_dataclass
class RequestLogTraceFlavorConfig(BaseTraceFlavorConfig):
    """Request log trace flavor: independent requests with token lengths only.

    Each row is a standalone request with input_length and output_length.
    No session structure, no corpus files, no prompt materialization.
    Supports CSV and JSONL trace files.
    """

    @classmethod
    def get_type(cls):
        return TraceFlavorType.REQUEST_LOG


@frozen_dataclass
class UntimedContentMultiTurnTraceFlavorConfig(BaseTraceFlavorConfig):
    """Untimed content multi-turn trace flavor: replay datasets with actual message content.

    Supports datasets like ShareGPT, LMSYS-Chat, etc. where each row
    contains a full conversation with actual text content.
    """

    conversation_column: str = field(
        "conversations", help="Column containing the list of conversation messages"
    )
    role_key: str = field("from", help="Key for the role field in each message dict")
    content_key: str = field(
        "value", help="Key for the content field in each message dict"
    )
    user_role_value: str = field("human", help="Role value indicating user messages")
    assistant_role_value: str = field(
        "gpt", help="Role value indicating assistant messages"
    )

    @classmethod
    def get_type(cls):
        return TraceFlavorType.UNTIMED_CONTENT_MULTI_TURN


@frozen_dataclass
class ShareGPTTraceFlavorConfig(BaseTraceFlavorConfig):
    """ShareGPT conversation trace flavor configuration.

    Reads ShareGPT-format conversations and uses assistant turn text
    as TTS input. Each assistant turn becomes a single-request session.
    """

    assistant_role: str = field(
        "gpt",
        help="Role name for assistant turns in the ShareGPT data "
        "(common values: 'gpt', 'assistant').",
    )
    min_tokens: int = field(
        20,
        help="Minimum input token count. Turns shorter than this are skipped. "
        "Ignored when min_chars/max_chars are set.",
    )
    max_tokens: int = field(
        100,
        help="Maximum input token count. Text is truncated to sampled length. "
        "Ignored when min_chars/max_chars are set.",
    )
    min_chars: int = field(
        -1,
        help="If >= 0, enables char-based input length control (mutually exclusive "
        "with min_tokens/max_tokens). Turns with fewer than this many chars are skipped.",
    )
    max_chars: int = field(
        -1,
        help="If >= 0, enables char-based input length control (mutually exclusive "
        "with min_tokens/max_tokens). Text is truncated to a char length sampled "
        "uniformly in [min_chars, max_chars].",
    )
    min_alpha_ratio: float = field(
        0.5,
        help="Minimum ratio of alphabetic characters to total non-space characters. "
        "Filters out junk entries like number sequences or code snippets. "
        "Set to 0.0 to disable.",
    )

    @property
    def use_chars(self) -> bool:
        return self.min_chars >= 0 and self.max_chars >= 0

    def __post_init__(self):
        one_set = (self.min_chars >= 0) != (self.max_chars >= 0)
        if one_set:
            raise ValueError(
                "min_chars and max_chars must both be set (>= 0) or both left "
                f"unset (-1); got min_chars={self.min_chars}, max_chars={self.max_chars}"
            )
        if self.use_chars:
            if self.min_chars > self.max_chars:
                raise ValueError(
                    f"min_chars ({self.min_chars}) must be <= max_chars ({self.max_chars})"
                )
        elif self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) must be <= max_tokens ({self.max_tokens})"
            )

    @classmethod
    def get_type(cls):
        return TraceFlavorType.SHAREGPT


@frozen_dataclass
class AudioTraceFlavorConfig(BaseTraceFlavorConfig):
    """Audio trace flavor configuration for STT benchmarking."""

    audio_dir: str = field(
        "",
        help="Optional base directory prepended to relative audio_file paths.",
    )
    target_duration_s: float | None = field(
        None,
        help=(
            "Optional per-session audio duration to stream. The trace must "
            "provide reference_word_timestamps so expected transcripts can be "
            "trimmed to the streamed prefix."
        ),
    )
    target_duration_spread_s: float | None = field(
        None,
        help=(
            "Hard half-width of a clipped-Gaussian duration spread around "
            "target_duration_s. Requires target_duration_s."
        ),
    )
    target_duration_sigma_s: float | None = field(
        None,
        help=(
            "Standard deviation of the clipped-Gaussian duration draw. Defaults "
            "to half the spread and must not exceed the spread."
        ),
    )

    @classmethod
    def get_type(cls):
        return TraceFlavorType.AUDIO

    def __post_init__(self):
        if self.target_duration_s is not None and self.target_duration_s <= 0:
            raise ValueError(
                "target_duration_s must be positive when set; "
                f"got {self.target_duration_s}"
            )
        if self.target_duration_spread_s is not None:
            if self.target_duration_s is None:
                raise ValueError("target_duration_spread_s requires target_duration_s")
            if not 0 < self.target_duration_spread_s < self.target_duration_s:
                raise ValueError(
                    "target_duration_spread_s must be in (0, "
                    f"target_duration_s); got {self.target_duration_spread_s} "
                    f"with target_duration_s={self.target_duration_s}"
                )
        if self.target_duration_sigma_s is not None:
            if self.target_duration_spread_s is None:
                raise ValueError(
                    "target_duration_sigma_s requires target_duration_spread_s"
                )
            if self.target_duration_sigma_s <= 0:
                raise ValueError(
                    "target_duration_sigma_s must be positive; got "
                    f"{self.target_duration_sigma_s}"
                )
            if self.target_duration_sigma_s > self.target_duration_spread_s:
                raise ValueError(
                    "target_duration_sigma_s must be <= "
                    "target_duration_spread_s; got sigma="
                    f"{self.target_duration_sigma_s} spread="
                    f"{self.target_duration_spread_s}"
                )


@frozen_dataclass
class SeedTTSTextTraceFlavorConfig(BaseTraceFlavorConfig):
    """Seed TTS eval text dataset configuration.

    Loads one text-only TTS request per dataset row. The default source is the
    English split of the Hugging Face Seed-TTS eval mirror.
    """

    dataset_name: str = field(
        "TwinkStart/Seed-TTS-Eval",
        help="Hugging Face dataset name used when local_path is empty.",
    )
    subset: str = field("en", help="Dataset subset/config name. Defaults to English.")
    split: str = field("train", help="Dataset split to load.")
    revision: str = field(
        "",
        help=(
            "Optional Hugging Face dataset revision (commit SHA or tag). "
            "When set, load_dataset pins to this revision so dataset drift "
            "cannot silently change the workload."
        ),
    )
    text_column: str = field(
        "text", help="Column containing the target TTS synthesis text."
    )
    id_column: str = field(
        "filename",
        help="Optional source row identifier column copied into request metadata.",
    )
    local_path: str = field(
        "",
        help="Optional local dataset path. Supports saved HF datasets and common "
        "data files such as JSON/JSONL, CSV, and Parquet.",
    )
    min_tokens: int = field(
        20,
        help="Minimum input word count. Rows with fewer words are skipped. "
        "Ignored when min_chars/max_chars are set.",
    )
    max_tokens: int = field(
        150,
        help="Maximum input word count. Text is truncated to a sampled word "
        "count in [min_tokens, max_tokens]. Ignored when min_chars/max_chars "
        "are set.",
    )
    min_chars: int = field(
        -1,
        help="If >= 0, enables char-based input length control. Rows with "
        "fewer chars are skipped.",
    )
    max_chars: int = field(
        -1,
        help="If >= 0, enables char-based input length control. Text is "
        "truncated to a sampled char count in [min_chars, max_chars].",
    )

    @property
    def use_chars(self) -> bool:
        return self.min_chars >= 0 and self.max_chars >= 0

    @classmethod
    def get_type(cls):
        return TraceFlavorType.SEED_TTS_TEXT

    def __post_init__(self):
        if not self.text_column:
            raise ValueError("SeedTTSTextTraceFlavorConfig.text_column is required.")
        if not self.local_path and not self.dataset_name:
            raise ValueError(
                "SeedTTSTextTraceFlavorConfig requires dataset_name or local_path."
            )
        one_char_bound_set = (self.min_chars >= 0) != (self.max_chars >= 0)
        if one_char_bound_set:
            raise ValueError(
                "min_chars and max_chars must both be set (>= 0) or both left "
                f"unset (-1); got min_chars={self.min_chars}, max_chars={self.max_chars}"
            )
        if self.use_chars:
            if self.min_chars > self.max_chars:
                raise ValueError(
                    f"min_chars ({self.min_chars}) must be <= max_chars ({self.max_chars})"
                )
        else:
            if self.min_tokens < 0 or self.max_tokens < 0:
                raise ValueError("min_tokens and max_tokens must be non-negative.")
            if self.min_tokens > self.max_tokens:
                raise ValueError(
                    f"min_tokens ({self.min_tokens}) must be <= max_tokens ({self.max_tokens})"
                )


# ----- Trace Session Generator Config -----


@frozen_dataclass
class TraceSessionGeneratorConfig(BaseSessionGeneratorConfig):
    """Trace-driven session generator configuration."""

    trace_file: str = field("", help="Path to the trace file (JSONL or CSV)")
    wrap_mode: bool = field(
        True, help="Whether to wrap/loop over the trace indefinitely"
    )
    flavor: BaseTraceFlavorConfig = field(
        default_factory=TimedSyntheticSessionTraceFlavorConfig,
        help="Trace flavor configuration.",
    )

    @classmethod
    def get_type(cls):
        return SessionGeneratorType.TRACE
