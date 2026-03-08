"""Untimed content multi-turn trace flavor generator.

Replays datasets with actual message content (ShareGPT, LMSYS-Chat, etc.).
Each row contains a full conversation; turns are split into individual
requests with pre-populated history.
"""

import json
import logging
from typing import Any, Dict, List

import pandas as pd

from veeksha.config.generator.session import (
    TraceSessionGeneratorConfig,
    UntimedContentMultiTurnTraceFlavorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.generator.session.trace.base_flavor import TraceFlavorGeneratorBase
from veeksha.logger import init_logger

logger = init_logger(__name__)


class UntimedContentMultiTurnTraceFlavorGenerator(TraceFlavorGeneratorBase):
    """Replay datasets with actual conversation content.

    Each row in the trace contains a list of messages (user/assistant pairs).
    The generator splits each conversation into turn-level requests,
    pre-populating ``request.history`` with prior messages. Edges use
    ``is_history_parent=False`` since history is pre-populated from the
    dataset rather than built dynamically by the traffic layer.
    """

    def __init__(
        self,
        config: TraceSessionGeneratorConfig,
        flavor_config: UntimedContentMultiTurnTraceFlavorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
    ):
        self._flavor_config: UntimedContentMultiTurnTraceFlavorConfig = flavor_config
        super().__init__(config, flavor_config, seed_manager, tokenizer_provider)

        # Auto-assign session_id if missing (each row = one conversation)
        if "session_id" not in self.trace_df.columns:
            self.trace_df["session_id"] = range(len(self.trace_df))

        # Log stats
        num_convos = len(self.trace_df)
        avg_turns = self._compute_avg_turns()
        logger.info(
            "Untimed content multi-turn trace loaded: %d conversations, avg %.1f turns",
            num_convos,
            avg_turns,
        )

    @property
    def required_columns(self) -> List[str]:
        return [self._flavor_config.conversation_column]

    def _compute_avg_turns(self) -> float:
        """Compute average number of user turns across conversations."""
        total_turns = 0
        col = self._flavor_config.conversation_column
        for _, row in self.trace_df.iterrows():
            messages = self._parse_messages(row[col])
            pairs = self._extract_turn_pairs(messages)
            total_turns += len(pairs)
        return total_turns / max(len(self.trace_df), 1)

    def _parse_messages(self, raw: Any) -> List[Dict[str, str]]:
        """Parse the conversation column into a list of message dicts."""
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, list):
            return []
        return raw

    def _extract_turn_pairs(self, messages: List[Dict[str, str]]) -> List[tuple]:
        """Extract (user_content, assistant_content) pairs from messages.

        Skips leading assistant messages, system messages, and trailing
        user messages without a response.
        """
        cfg = self._flavor_config
        pairs: List[tuple] = []

        i = 0
        # Skip leading non-user messages
        while (
            i < len(messages) and messages[i].get(cfg.role_key) != cfg.user_role_value
        ):
            i += 1

        while i < len(messages):
            msg = messages[i]
            role = msg.get(cfg.role_key)

            if role == cfg.user_role_value:
                content = msg.get(cfg.content_key, "")
                # Look for the next assistant message
                if (
                    i + 1 < len(messages)
                    and messages[i + 1].get(cfg.role_key) == cfg.assistant_role_value
                ):
                    assistant_content = messages[i + 1].get(cfg.content_key, "")
                    pairs.append((content, assistant_content))
                    i += 2
                else:
                    # Trailing user message without response — skip
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Skipping trailing user message without assistant response"
                        )
                    i += 1
            else:
                # Skip non-user/non-assistant messages (system, etc.)
                i += 1

        return pairs

    def prepare_session(self, group: pd.DataFrame) -> Session:
        """Create a multi-turn session from conversation data."""
        session_id = self._next_session_id()
        cfg = self._flavor_config
        col = cfg.conversation_column

        # Collect all turn pairs from all rows in the group
        all_pairs: List[tuple] = []
        for _, row_series in group.iterrows():
            raw = row_series[col]
            messages = self._parse_messages(raw)
            pairs = self._extract_turn_pairs(messages)
            all_pairs.extend(pairs)

        if not all_pairs:
            logger.warning(
                "Session %d has no valid turn pairs, creating minimal session",
                session_id,
            )
            # Create a single empty request as fallback
            request = self._create_text_request(
                node_id=0,
                prompt_text="",
                target_output_tokens=1,
                wait_after_ready=0.0,
            )
            session_graph = self._build_linear_session_graph(1, [0.0])
            return Session(
                id=session_id,
                session_graph=session_graph,
                requests={0: request},
            )

        requests = {}
        history: List[Dict[str, str]] = []

        for i, (user_text, assistant_text) in enumerate(all_pairs):
            # Compute token counts
            assistant_tokens = len(self.tokenizer.encode(assistant_text))

            # Compute target_prompt_tokens: history + current user message
            history_text = " ".join(msg["content"] for msg in history)
            if history_text:
                full_prompt_text = history_text + " " + user_text
            else:
                full_prompt_text = user_text
            target_prompt_tokens = len(self.tokenizer.encode(full_prompt_text))

            request = self._create_text_request(
                node_id=i,
                prompt_text=user_text,
                target_output_tokens=assistant_tokens,
                wait_after_ready=0.0,
                parent_node=i - 1 if i > 0 else None,
                target_prompt_tokens=target_prompt_tokens,
            )
            # Pre-populate history from the dataset
            request.history = list(history)

            requests[i] = request

            # Accumulate history for next turn
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": assistant_text})

        session_graph = self._build_linear_session_graph(
            len(requests),
            [0.0] * len(requests),
            is_history_parent=False,
        )

        return Session(
            id=session_id,
            session_graph=session_graph,
            requests=requests,
        )

    def wrap(self) -> pd.DataFrame:
        """Wrap trace for new epoch — shuffle and increment session IDs."""
        df = self.trace_df.copy()
        df["session_id"] = df["session_id"] + df["session_id"].max() + 1
        return self._shuffle_sessions(df)
