"""ShareGPT trace flavor generator for TTS benchmarking.

Reads ShareGPT-format conversation data and uses assistant turn text
as TTS request input. Each assistant turn becomes its own single-request
session. Input text is truncated to a length sampled uniformly between
min_tokens..max_tokens (token mode, default) or min_chars..max_chars
(char mode, when chars are set on the flavor config).
"""

import json
import os
from typing import Any, Dict, List

import pandas as pd

from veeksha.config.generator.session import (
    ShareGPTTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.generator.session.trace.base_flavor import (
    TraceFlavorGeneratorBase,
)
from veeksha.logger import init_logger

logger = init_logger(__name__)


def _load_sharegpt_file(path: str) -> List[Dict[str, Any]]:
    """Load a ShareGPT file (JSON array or JSONL)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try JSON array first
    if content.startswith("["):
        return json.loads(content)

    # Fall back to JSONL
    conversations = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            conversations.append(json.loads(line))
    return conversations


def _alpha_ratio(text: str) -> float:
    """Return ratio of alphabetic characters to total non-space characters."""
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 0.0
    return sum(1 for c in non_space if c.isalpha()) / len(non_space)


def _flatten_to_dataframe(
    conversations: List[Dict[str, Any]],
    assistant_role: str,
    tokenizer: Any,
    min_length: int,
    use_chars: bool,
    min_alpha_ratio: float = 0.5,
) -> pd.DataFrame:
    """Convert ShareGPT conversations to a flat DataFrame of assistant turns.

    Each assistant turn gets its own session_id (1 request per session).
    Turns shorter than min_length are skipped, where the unit is chars
    (use_chars=True) or tokens (use_chars=False).
    Turns with alpha ratio below min_alpha_ratio are skipped (filters out
    number sequences, code snippets, etc.).
    Returns DataFrame with columns: session_id, text, token_count, char_count
    """
    unit = "chars" if use_chars else "tokens"
    rows = []
    session_id = 0
    skipped_length = 0
    skipped_alpha = 0
    for conv in conversations:
        turns = conv.get("conversations", [])
        for turn in turns:
            role = turn.get("from", "")
            if role == assistant_role:
                text = turn.get("value", "").strip()
                if not text:
                    continue
                # Filter out junk text (number sequences, code, etc.)
                if min_alpha_ratio > 0 and _alpha_ratio(text) < min_alpha_ratio:
                    skipped_alpha += 1
                    continue
                char_count = len(text)
                if use_chars:
                    # Skip token encode in char mode -- it's the hot path filter.
                    if char_count < min_length:
                        skipped_length += 1
                        continue
                    token_count = -1
                else:
                    token_count = len(tokenizer.encode(text))
                    if token_count < min_length:
                        skipped_length += 1
                        continue
                rows.append(
                    {
                        "session_id": session_id,
                        "text": text,
                        "token_count": token_count,
                        "char_count": char_count,
                    }
                )
                session_id += 1

    if skipped_length:
        logger.info(
            "Skipped %d assistant turns shorter than %d %s",
            skipped_length,
            min_length,
            unit,
        )
    if skipped_alpha:
        logger.info(
            "Skipped %d assistant turns with alpha ratio below %.2f "
            "(number sequences, code, etc.)",
            skipped_alpha,
            min_alpha_ratio,
        )

    if not rows:
        min_key = "min_chars" if use_chars else "min_tokens"
        raise ValueError(
            f"No assistant turns found with role='{assistant_role}' "
            f"and >= {min_length} {unit} and alpha_ratio >= {min_alpha_ratio}. "
            f"Check 'assistant_role', '{min_key}', and 'min_alpha_ratio' config."
        )

    return pd.DataFrame(rows)


class ShareGPTTraceFlavorGenerator(TraceFlavorGeneratorBase):
    """Trace flavor that reads ShareGPT conversations for TTS benchmarking.

    Each assistant turn becomes a single-request session. Input text is
    truncated to a length sampled uniformly between min_tokens..max_tokens
    (token mode) or min_chars..max_chars (char mode, when chars are set).
    Turns shorter than the minimum are skipped entirely.
    """

    def __init__(
        self,
        config: TraceSessionGeneratorConfig,
        flavor_config: ShareGPTTraceFlavorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
    ):
        # We override the base __init__ because ShareGPT uses a different
        # file format (JSON array / JSONL with "conversations" field) rather
        # than the standard JSONL with session_id/input_length columns.
        self.config = config
        self.flavor_config = flavor_config
        self.seed_manager = seed_manager
        self.tokenizer_provider = tokenizer_provider
        from veeksha.types import ChannelModality

        self.tokenizer = tokenizer_provider.for_modality(ChannelModality.TEXT)

        if not os.path.exists(config.trace_file):
            raise FileNotFoundError(f"Trace file not found: {config.trace_file}")

        raw_conversations = _load_sharegpt_file(config.trace_file)
        logger.info(
            "Loaded %d conversations from %s",
            len(raw_conversations),
            config.trace_file,
        )

        use_chars = flavor_config.use_chars
        min_length = flavor_config.min_chars if use_chars else flavor_config.min_tokens
        self.trace_df = _flatten_to_dataframe(
            raw_conversations,
            flavor_config.assistant_role,
            self.tokenizer,
            min_length,
            use_chars,
            flavor_config.min_alpha_ratio,
        )
        logger.info(
            "Extracted %d assistant turns (1 session each, min_%s=%d)",
            len(self.trace_df),
            "chars" if use_chars else "tokens",
            min_length,
        )

        # wrapping state (same as base)
        self._num_wraps = 0
        self._session_groups = None
        self._current_session_id = 0
        self._current_request_id = 0
        self._rng = seed_manager.random("trace_shuffling")
        self._length_rng = seed_manager.random("sharegpt_input_length")

    @property
    def required_columns(self) -> List[str]:
        return ["session_id", "text"]

    def _truncate_to_tokens(self, text: str, target_tokens: int) -> str:
        """Truncate text to exactly target_tokens by encoding and decoding."""
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) <= target_tokens:
            return text
        truncated_ids = token_ids[:target_tokens]
        return self.tokenizer.decode(truncated_ids)

    def prepare_session(self, group: pd.DataFrame) -> Session:
        """Convert a single assistant turn into a single-request Session."""
        session_id = self._next_session_id()
        row = group.iloc[0]
        text = str(row["text"])

        if self.flavor_config.use_chars:
            target_chars = self._length_rng.randint(
                self.flavor_config.min_chars,
                self.flavor_config.max_chars,
            )
            if len(text) > target_chars:
                text = text[:target_chars]
            actual_tokens = len(self.tokenizer.encode(text))
        else:
            target_tokens = self._length_rng.randint(
                self.flavor_config.min_tokens,
                self.flavor_config.max_tokens,
            )
            text = self._truncate_to_tokens(text, target_tokens)
            actual_tokens = len(self.tokenizer.encode(text))

        request = self._create_text_request(
            node_id=0,
            prompt_text=text,
            target_output_tokens=1,  # TTS doesn't use output tokens
            wait_after_ready=0.0,
            parent_nodes=[],
            target_prompt_tokens=actual_tokens,
        )
        graph = self._build_linear_session_graph(1, [0.0])
        return Session(
            id=session_id,
            session_graph=graph,
            requests={0: request},
        )

    def wrap(self) -> pd.DataFrame:
        """Wrap trace for new epoch with shuffled session order."""
        df = self.trace_df.copy()
        max_sid = int(df["session_id"].max()) if not df.empty else 0
        df["session_id"] = df["session_id"] + max_sid + 1
        return self._shuffle_sessions(df)
