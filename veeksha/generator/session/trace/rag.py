"""RAG trace flavor generator with warmup support."""

from typing import Any, List, Optional

import pandas as pd

from veeksha.config.generator.session import (
    RAGTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.generator.session.trace.base_flavor import (
    TraceFlavorGeneratorBase,
)


class RAGTraceFlavorGenerator(TraceFlavorGeneratorBase):
    """RAG trace flavor generator with warmup sessions per document."""

    def __init__(
        self,
        config: TraceSessionGeneratorConfig,
        flavor_config: RAGTraceFlavorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
    ):
        super().__init__(config, flavor_config, seed_manager, tokenizer_provider)
        self.flavor_config = flavor_config
        self._warmup_sessions: Optional[List[Session]] = None

        doc_counts = self.trace_df["doc_id"].value_counts()

        requested_docs = max(int(flavor_config.num_documents), 1)
        self._top_doc_ids: List[Any] = doc_counts.nlargest(
            requested_docs
        ).index.tolist()
        self._warmup_count = len(self._top_doc_ids)

        filtered_df = (
            self.trace_df[self.trace_df["doc_id"].isin(self._top_doc_ids)]
            .reset_index(drop=True)
            .copy()
        )
        if filtered_df.empty:
            raise ValueError("No trace rows remain after filtering to top documents. ")

        session_start = self._warmup_count
        filtered_df["session_id"] = range(
            session_start, session_start + len(filtered_df)
        )

        self._warmup_source_df = self._build_warmup_source(filtered_df)
        # shuffle operational trace to emulate historical behavior
        self.trace_df = self._shuffle_sessions(filtered_df)
        # ensure generated sessions follow warmup id range
        self._session_offset = session_start
        self._current_session_id = self._session_offset

    @property
    def required_columns(self) -> List[str]:
        return [
            "doc_id",
            "prompt_text",
            "input_length",
            "output_length",
        ]

    def _validate_trace(self):
        """Ensure the trace contains required columns and session ids."""
        super()._validate_trace()

        if "session_id" not in self.trace_df.columns:
            self.trace_df = self.trace_df.reset_index(drop=True)
            self.trace_df["session_id"] = range(len(self.trace_df))

    def get_warmup_sessions(self) -> List[Session]:
        """Generate warmup sessions-one per selected document."""
        if self._warmup_sessions is not None:
            return self._warmup_sessions

        sessions: List[Session] = []
        for session_id, (_, row) in enumerate(self._warmup_source_df.iterrows()):
            sessions.append(self._build_single_request_session(session_id, row))

        self._warmup_sessions = sessions
        # reshuffle main trace before benchmark sessions start
        self.trace_df = self._shuffle_sessions(self.trace_df)
        self._session_groups = None
        return self._warmup_sessions

    def prepare_session(self, group: pd.DataFrame) -> Session:
        """Prepare a single-request RAG session from the trace row."""
        row = group.iloc[0]
        session_id = self._next_session_id()
        return self._build_single_request_session(session_id, row)

    def wrap(self) -> pd.DataFrame:
        """Wrap trace for new epoch with refreshed session order."""
        df = self.trace_df.copy()
        max_sid = int(df["session_id"].max()) if not df.empty else self._session_offset
        df["session_id"] = df["session_id"] + max_sid + 1
        return self._shuffle_sessions(df)

    def _build_warmup_source(self, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """Capture a deterministic row per doc_id for warmup sessions."""
        rows = []
        for doc_id in self._top_doc_ids:
            doc_rows = filtered_df[filtered_df["doc_id"] == doc_id]
            if not doc_rows.empty:
                rows.append(doc_rows.iloc[0])
        if rows:
            warmup_df = pd.DataFrame(rows).reset_index(drop=True)
        else:
            warmup_df = pd.DataFrame(columns=filtered_df.columns)

        warmup_df["session_id"] = range(len(warmup_df))
        return warmup_df

    def _build_single_request_session(self, session_id: int, row: pd.Series) -> Session:
        """Convert a single trace row into a Veeksha Session."""
        wait_time = 0.0
        request = self._create_text_request(
            node_id=0,
            prompt_text=str(row["prompt_text"]),
            target_output_tokens=int(row["output_length"]),
            wait_after_ready=wait_time,
            parent_node=None,
            target_prompt_tokens=int(row["input_length"]),
        )
        session_graph = self._build_linear_session_graph(1, [wait_time])
        return Session(
            id=session_id,
            session_graph=session_graph,
            requests={0: request},
        )
