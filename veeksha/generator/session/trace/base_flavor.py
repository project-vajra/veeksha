"""Base class for trace flavor generators."""

import os
from abc import abstractmethod
from typing import Iterator, List, Optional, cast

import pandas as pd

from veeksha.config.generator.session import (
    BaseTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import (
    SessionEdge,
    SessionGraph,
    SessionNode,
    add_edge,
    add_node,
)
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.types import ChannelModality


class TraceFlavorGeneratorBase:
    """Base class for trace flavor generators.

    Subclasses implement flavor-specific logic for:
    - Required trace columns validation
    - Prompt preparation from trace rows
    - Wrapping/epoch logic for looping
    """

    def __init__(
        self,
        config: TraceSessionGeneratorConfig,
        flavor_config: BaseTraceFlavorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
    ):
        self.config = config
        self.flavor_config = flavor_config
        self.seed_manager = seed_manager
        self.tokenizer_provider = tokenizer_provider
        self.tokenizer = tokenizer_provider.for_modality(ChannelModality.TEXT)

        self._validate_trace_exists(config.trace_file)
        self.trace_df = pd.read_json(config.trace_file, lines=True)
        self._validate_trace()

        # wrapping state
        self._num_wraps = 0
        self._session_groups: Optional[Iterator] = None
        self._current_session_id = 0
        self._current_request_id = 0

        self._rng = seed_manager.random("trace_shuffling")

    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """Columns required in the trace DataFrame."""

    @abstractmethod
    def prepare_session(self, group: pd.DataFrame) -> Session:
        """Prepare a Session from a trace session group."""

    @abstractmethod
    def wrap(self) -> pd.DataFrame:
        """Wrap the trace for a new epoch."""

    def get_warmup_sessions(self) -> List[Session]:
        """Return warmup sessions. Default empty, override for RAG."""
        return []

    def _validate_trace_exists(self, trace_file: str):
        """Validate that trace file exists."""
        if not os.path.exists(trace_file):
            raise FileNotFoundError(f"Trace file not found: {trace_file}")

    def _validate_trace(self):
        """Validate that required columns exist in trace."""
        for col in self.required_columns:
            if col not in self.trace_df.columns:
                raise ValueError(
                    f"Trace missing required column '{col}'. "
                    f"Required: {self.required_columns}"
                )

    def _get_session_groups(self) -> Iterator:
        """Get iterator over session groups."""
        return iter(self.trace_df.groupby("session_id"))

    def _next_session_id(self) -> int:
        """Get next global session ID."""
        sid = self._current_session_id
        self._current_session_id += 1
        return sid

    def _next_request_id(self) -> int:
        """Get next global request ID."""
        rid = self._current_request_id
        self._current_request_id += 1
        return rid

    def generate_session(self) -> Session:
        """Generate the next session from the trace."""
        if self._session_groups is None:
            self._session_groups = self._get_session_groups()

        try:
            _, group = next(self._session_groups)
            return self.prepare_session(group)
        except StopIteration:
            if self.config.wrap_mode:
                self.trace_df = self.wrap()
                self._num_wraps += 1
                self._session_groups = self._get_session_groups()
                _, group = next(self._session_groups)
                return self.prepare_session(group)
            else:
                raise StopIteration("Trace exhausted and wrap_mode is False")

    def capacity(self) -> int:
        """Return -1 (unbounded) if wrap mode, else session count."""
        if self.config.wrap_mode:
            return -1
        return len(self.trace_df.groupby("session_id"))

    def _build_linear_session_graph(
        self, num_requests: int, wait_times: List[float]
    ) -> SessionGraph:
        """Build a linear session graph (1→2→3...)."""
        graph = SessionGraph()
        for i in range(num_requests):
            wait = wait_times[i] if i < len(wait_times) else 0.0
            add_node(graph, SessionNode(id=i, wait_after_ready=wait))
            if i > 0:
                add_edge(graph, SessionEdge(src=i - 1, dst=i, is_history_parent=True))
        return graph

    def _create_text_request(
        self,
        node_id: int,
        prompt_text: str,
        target_output_tokens: int,
        wait_after_ready: float,
        parent_node: Optional[int] = None,
        target_prompt_tokens: Optional[int] = None,
    ) -> Request:
        """Create a text-only Request and attach output spec."""

        channels = {
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text=prompt_text,
                target_prompt_tokens=target_prompt_tokens,
            )
        }
        session_context = {
            "node_id": node_id,
            "wait_after_ready": wait_after_ready,
            "parent_nodes": [parent_node] if parent_node is not None else [],
            "history_parent": parent_node,
        }
        requested_output = RequestedOutputSpec(
            text=TextOutputSpec(target_tokens=target_output_tokens)
        )
        return Request(
            id=self._next_request_id(),
            channels=channels,  # type: ignore
            session_context=session_context,
            requested_output=requested_output,
        )

    def _shuffle_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shuffle session order in DataFrame."""
        sid_order = df["session_id"].unique().tolist()
        self._rng.shuffle(sid_order)
        df_shuffled = pd.concat(
            [df[df["session_id"] == sid] for sid in sid_order]
        ).reset_index(drop=True)
        return cast(pd.DataFrame, df_shuffled)
