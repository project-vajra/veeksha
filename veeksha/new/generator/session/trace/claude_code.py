"""Context-Cached trace flavor generator."""

from pathlib import Path
from typing import List

import pandas as pd

from veeksha.new.config.generator.session import (
    ClaudeCodeTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.new.core.seeding import SeedManager
from veeksha.new.core.session import Session
from veeksha.new.core.tokenizer import TokenizerProvider
from veeksha.new.generator.session.trace.base_flavor import TraceFlavorGeneratorBase
from veeksha.new.generator.session.trace.prompt_builder import TracePromptBuilder


class ClaudeCodeTraceFlavorGenerator(TraceFlavorGeneratorBase):
    """Context-Cached trace flavor generator.

    Generates unique prefix tokens per session to enable KV-cache sharing.
    Sessions share a common prefix but have unique suffixes.
    """

    def __init__(
        self,
        config: TraceSessionGeneratorConfig,
        flavor_config: ClaudeCodeTraceFlavorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
    ):
        super().__init__(config, flavor_config, seed_manager, tokenizer_provider)
        self.flavor_config = flavor_config

        self.prompt_builder = TracePromptBuilder(
            tokenizer=self.tokenizer,
            seed_manager=seed_manager.child("prompt_builder"),
            corpus_file=(
                Path(flavor_config.corpus_file) if flavor_config.corpus_file else None
            ),
        )
        self._session_seed_rng = seed_manager.random("cc_session_seeds")
        self._wrap_rng = seed_manager.random("cc_wrapping")

    @property
    def required_columns(self) -> List[str]:
        return [
            "session_id",
            "input_length",
            "output_length",
        ]

    def prepare_session(self, group: pd.DataFrame) -> Session:
        """Prepare session with unique prefix for KV-cache."""
        session_id = self._next_session_id()
        requests = {}
        wait_times: List[float] = []

        # unique seed for this session's prefix
        session_seed = self._session_seed_rng.getrandbits(32)

        for i, (_, row) in enumerate(group.iterrows()):
            input_length = int(row["input_length"])
            output_length = int(row["output_length"])

            prompt_text = self.prompt_builder.generate_unique_prompt(
                num_tokens=input_length,
                page_size=self.flavor_config.page_size,
                seed=session_seed,
            )

            wait_time_val = row.get("wait_after_previous_response_s")
            if wait_time_val is None or pd.isna(wait_time_val):
                wait_time = 0.0
            else:
                wait_time = float(wait_time_val)
            wait_times.append(wait_time)

            request = self._create_text_request(
                node_id=i,
                prompt_text=prompt_text,
                target_output_tokens=output_length,
                wait_after_ready=wait_time,
                parent_node=i - 1 if i > 0 else None,
                target_prompt_tokens=input_length,
            )
            requests[i] = request

        session_graph = self._build_linear_session_graph(len(requests), wait_times)

        return Session(
            id=session_id,
            session_graph=session_graph,
            requests=requests,
        )

    def wrap(self) -> pd.DataFrame:
        """Wrap trace for new epoch with new session seeds."""
        df = self.trace_df.copy()
        df["session_id"] = df["session_id"] + df["session_id"].max() + 1
        return self._shuffle_sessions(df)
