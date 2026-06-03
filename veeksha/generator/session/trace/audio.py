"""Audio trace flavor generator for STT benchmarking.

Reads a JSONL trace file where each line contains:
    {"session_id": 0, "audio_file": "/path/to/audio.wav"}

Optionally, each line may include an ``expected_transcript`` field for
WER evaluation.

Each row becomes a single-request session with an AUDIO channel
containing the file path in ``AudioChannelRequestContent.input_audio``.
"""

import os
from typing import Any, List

import numpy as np
import pandas as pd

from veeksha.config.generator.session import (
    AudioTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.request import Request
from veeksha.core.request_content import AudioChannelRequestContent
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.generator.session.trace.base_flavor import (
    TraceFlavorGeneratorBase,
)
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

logger = init_logger(__name__)


class AudioTraceFlavorGenerator(TraceFlavorGeneratorBase):
    """Trace flavor that feeds audio file paths for STT benchmarking.

    Each row in the JSONL becomes a single-request session whose AUDIO
    channel points to the audio file on disk.
    """

    def __init__(
        self,
        config: TraceSessionGeneratorConfig,
        flavor_config: AudioTraceFlavorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
    ):
        self.config = config
        self.flavor_config = flavor_config
        self.seed_manager = seed_manager
        self.tokenizer_provider = tokenizer_provider
        self.tokenizer = tokenizer_provider.for_modality(ChannelModality.TEXT)

        if not os.path.exists(config.trace_file):
            raise FileNotFoundError(f"Trace file not found: {config.trace_file}")

        self.trace_df = pd.read_json(config.trace_file, lines=True)
        self._validate_trace()

        # Ground truth is mandatory; fail at load, not silently per request.
        col = self.trace_df["expected_transcript"]
        missing = col.isna() | (col.astype(str).str.strip() == "")
        if missing.any():
            examples = self.trace_df.loc[missing, "audio_file"].head(3).tolist()
            raise ValueError(
                f"{int(missing.sum())} audio trace row(s) missing "
                f"expected_transcript (e.g. {examples})."
            )

        # Resolve relative paths against audio_dir when provided, otherwise
        # against the manifest directory so manifests are portable.
        trace_dir = os.path.dirname(os.path.abspath(config.trace_file))
        audio_base = flavor_config.audio_dir or trace_dir
        if not os.path.isabs(audio_base):
            audio_base = os.path.join(trace_dir, audio_base)
        self.trace_df["audio_file"] = self.trace_df["audio_file"].apply(
            lambda p: p if os.path.isabs(str(p)) else os.path.join(audio_base, str(p))
        )
        missing_audio = ~self.trace_df["audio_file"].apply(os.path.exists)
        if missing_audio.any():
            examples = self.trace_df.loc[missing_audio, "audio_file"].head(3).tolist()
            raise FileNotFoundError(
                f"{int(missing_audio.sum())} audio trace file(s) missing "
                f"(e.g. {examples})."
            )

        logger.info(
            "Loaded %d audio sessions from %s",
            len(self.trace_df.groupby("session_id")),
            config.trace_file,
        )

        # wrapping state
        self._num_wraps = 0
        self._session_groups = None
        self._current_session_id = 0
        self._current_request_id = 0
        self._rng = seed_manager.random("trace_shuffling")

    @property
    def required_columns(self) -> List[str]:
        return ["session_id", "audio_file", "expected_transcript"]

    def prepare_session(self, group: pd.DataFrame) -> Session:
        """Convert a single trace row into a single-request Session."""
        session_id = self._next_session_id()
        row = group.iloc[0]
        audio_file = str(row["audio_file"])

        channels = {
            ChannelModality.AUDIO: AudioChannelRequestContent(
                input_audio=audio_file,
            )
        }

        metadata = self._row_metadata(row)

        request = Request(
            id=self._next_request_id(),
            channels=channels,
            metadata=metadata,
            session_context={
                "node_id": 0,
                "wait_after_ready": 0.0,
                "parent_nodes": [],
                "history_parent": None,
            },
        )
        graph = self._build_linear_session_graph(1, [0.0])
        return Session(
            id=session_id,
            session_graph=graph,
            requests={0: request},
        )

    def _row_metadata(self, row: pd.Series) -> dict:
        """Pass manifest metadata through to the STT client/evaluator."""
        metadata: dict[str, Any] = {}
        for column in self.trace_df.columns:
            if column in {"session_id", "audio_file"}:
                continue
            value = row[column]
            if not isinstance(value, (dict, list)) and pd.isna(value):
                continue
            if isinstance(value, np.generic):
                value = value.item()
            metadata[column] = value
        metadata["expected_transcript"] = str(metadata["expected_transcript"])
        return metadata

    def wrap(self) -> pd.DataFrame:
        """Wrap trace for new epoch with shuffled session order."""
        df = self.trace_df.copy()
        max_sid = int(df["session_id"].max()) if not df.empty else 0
        df["session_id"] = df["session_id"] + max_sid + 1
        return self._shuffle_sessions(df)
