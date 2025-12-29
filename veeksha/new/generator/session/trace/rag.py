"""RAG trace flavor generator with warmup support."""

from typing import List, Optional

import pandas as pd

from veeksha.new.config.generator.session import (
    RAGTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.new.core.seeding import SeedManager
from veeksha.new.core.session import Session
from veeksha.new.core.tokenizer import TokenizerProvider
from veeksha.new.generator.session.trace.base_flavor import TraceFlavorGeneratorBase


class RAGTraceFlavorGenerator(TraceFlavorGeneratorBase):
    """RAG trace flavor generator.

    Supports document-based RAG traces with warmup capability.
    Can filter to top N documents and generate warmup sessions.
    """

    def __init__(
        self,
        config: TraceSessionGeneratorConfig,
        flavor_config: RAGTraceFlavorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
    ):
        super().__init__(config, flavor_config, seed_manager, tokenizer_provider)
        self.flavor_config = flavor_config

        # Filter to top N documents if specified
        if flavor_config.num_documents > 0 and "title" in self.trace_df.columns:
            top_docs = (
                self.trace_df.groupby("title")
                .size()
                .nlargest(flavor_config.num_documents)
                .index.tolist()
            )
            self.trace_df = self.trace_df[
                self.trace_df["title"].isin(top_docs)
            ].reset_index(drop=True)

        self._wrap_rng = seed_manager.random("rag_wrapping")
        self._warmup_sessions: Optional[List[Session]] = None

    @property
    def required_columns(self) -> List[str]:
        return [
            "session_id",
            "input_length",
            "output_length",
        ]

    def get_warmup_sessions(self) -> List[Session]:
        """Generate warmup sessions - one per unique document."""
        if self._warmup_sessions is not None:
            return self._warmup_sessions

        self._warmup_sessions = []

        if "title" not in self.trace_df.columns:
            return self._warmup_sessions

        # Create one session per unique document
        for title in self.trace_df["title"].unique():
            doc_rows = self.trace_df[self.trace_df["title"] == title]
            if len(doc_rows) == 0:
                continue

            # Take the first request from this document as warmup
            first_row = doc_rows.iloc[0]
            session_id = self._next_session_id()

            # Create a simple warmup request
            input_length = int(first_row.get("input_length", 100))
            prompt_text = f"Document warmup: {title[:100]}" + " padding" * (
                input_length // 10
            )

            request = self._create_text_request(
                node_id=0,
                prompt_text=prompt_text,
                target_output_tokens=10,  # Minimal output for warmup
                wait_after_ready=0.0,
                parent_node=None,
                target_prompt_tokens=input_length,
            )

            session_graph = self._build_linear_session_graph(1, [0.0])

            session = Session(
                id=session_id,
                session_graph=session_graph,
                requests={0: request},
            )
            self._warmup_sessions.append(session)

        return self._warmup_sessions

    def prepare_session(self, group: pd.DataFrame) -> Session:
        """Prepare session from RAG trace."""
        session_id = self._next_session_id()
        requests = {}
        wait_times: List[float] = []

        for i, (_, row) in enumerate(group.iterrows()):
            input_length = int(row["input_length"])
            output_length = int(row["output_length"])

            # Build prompt from trace data
            title = row.get("title", "document")
            prompt_text = f"Query for {title}: " + " padding" * (input_length // 10)

            # Get wait time
            wait_time_val = row.get("wait_after_previous_response_s")
            if wait_time_val is None or pd.isna(wait_time_val):
                wait_time = 0.0
            else:
                wait_time = float(wait_time_val)
            wait_times.append(wait_time)

            # Create request
            request = self._create_text_request(
                node_id=i,
                prompt_text=prompt_text,
                target_output_tokens=output_length,
                wait_after_ready=wait_time,
                parent_node=i - 1 if i > 0 else None,
                target_prompt_tokens=input_length,
            )
            requests[i] = request

        # Build session graph
        session_graph = self._build_linear_session_graph(len(requests), wait_times)

        return Session(
            id=session_id,
            session_graph=session_graph,
            requests=requests,
        )

    def wrap(self) -> pd.DataFrame:
        """Wrap trace for new epoch."""
        df = self.trace_df.copy()

        # Increment session IDs
        df["session_id"] = df["session_id"] + df["session_id"].max() + 1

        # Shuffle session order
        return self._shuffle_sessions(df)
