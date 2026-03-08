from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
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
    ChannelModality,
    SessionGeneratorType,
    SessionGraphType,
    TraceFlavorType,
)


@frozen_dataclass(allow_from_file=True)
class BaseSessionGeneratorConfig(BasePolyConfig):
    pass


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
        metadata={
            "help": f"The generator for the session graphs. {SessionGraphType.help_str()}"
        },
    )
    channels: list[BaseChannelGeneratorConfig] = field(
        default_factory=lambda: [TextChannelGeneratorConfig()],
        metadata={
            "help": f"The modality channels for the input content of each request. {ChannelModality.help_str()}"
        },
    )
    output_spec: OutputSpecConfig = field(
        default_factory=OutputSpecConfig,
        metadata={
            "help": "Specification for expected output from the model, for supported modalities (e.g., output token length, image count)."
        },
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
        metadata={"help": "The lm-eval tasks to evaluate the model on."},
    )
    num_fewshot: int = field(
        default=1,
        metadata={"help": "The number of fewshot examples to use for the tasks."},
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
class TimedSyntheticMultiTurnTraceFlavorConfig(BaseTraceFlavorConfig):
    """Timed synthetic multi-turn trace flavor configuration with context caching."""

    corpus_file: str = field(
        default="traces/corpus.txt",
        metadata={"help": "Path to corpus file for prompt padding"},
    )
    page_size: int = field(
        default=16,
        metadata={"help": "Number of unique tokens per session prefix"},
    )

    @classmethod
    def get_type(cls):
        return TraceFlavorType.TIMED_SYNTHETIC_MULTI_TURN


@frozen_dataclass
class SharedPrefixTraceFlavorConfig(BaseTraceFlavorConfig):
    """Shared-prefix trace flavor configuration with hash-based content sharing."""

    corpus_file: str = field(
        default="traces/corpus.txt",
        metadata={"help": "Path to corpus file for prompt padding"},
    )
    block_size: int = field(
        default=512,
        metadata={
            "help": "Number of tokens per hash id block. Only used for hash ids of first-in-session requests."
        },
    )

    @classmethod
    def get_type(cls):
        return TraceFlavorType.SHARED_PREFIX


@frozen_dataclass
class RAGTraceFlavorConfig(BaseTraceFlavorConfig):
    """RAG trace flavor configuration."""

    num_documents: int = field(
        default=10,
        metadata={"help": "Number of top documents to include for warmup"},
    )

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
        default="conversations",
        metadata={"help": "Column containing the list of conversation messages"},
    )
    role_key: str = field(
        default="from",
        metadata={"help": "Key for the role field in each message dict"},
    )
    content_key: str = field(
        default="value",
        metadata={"help": "Key for the content field in each message dict"},
    )
    user_role_value: str = field(
        default="human",
        metadata={"help": "Role value indicating user messages"},
    )
    assistant_role_value: str = field(
        default="gpt",
        metadata={"help": "Role value indicating assistant messages"},
    )

    @classmethod
    def get_type(cls):
        return TraceFlavorType.UNTIMED_CONTENT_MULTI_TURN


# ----- Trace Session Generator Config -----


@frozen_dataclass
class TraceSessionGeneratorConfig(BaseSessionGeneratorConfig):
    """Trace-driven session generator configuration."""

    trace_file: str = field(
        default="",
        metadata={"help": "Path to the trace file (JSONL or CSV)"},
    )
    wrap_mode: bool = field(
        default=True,
        metadata={"help": "Whether to wrap/loop over the trace indefinitely"},
    )
    flavor: BaseTraceFlavorConfig = field(
        default_factory=TimedSyntheticMultiTurnTraceFlavorConfig,
        metadata={"help": f"Trace flavor configuration. {TraceFlavorType.help_str()}"},
    )

    @classmethod
    def get_type(cls):
        return SessionGeneratorType.TRACE
