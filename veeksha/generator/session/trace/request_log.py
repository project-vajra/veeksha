"""Request log trace flavor generator.

Handles simple traces where each row is an independent request
with just input_length and output_length. No session structure,
no corpus files, no prompt materialization.
"""

from typing import List

import pandas as pd

from veeksha.config.generator.session import (
    RequestLogTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.generator.session.trace.base_flavor import TraceFlavorGeneratorBase
from veeksha.logger import init_logger

logger = init_logger(__name__)


class RequestLogTraceFlavorGenerator(TraceFlavorGeneratorBase):
    """Request log trace flavor generator.

    Each row becomes an independent single-request session.
    Only requires input_length and output_length columns.
    Prompts are generated as random token sequences.
    """

    def __init__(
        self,
        config: TraceSessionGeneratorConfig,
        flavor_config: RequestLogTraceFlavorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
    ):
        super().__init__(config, flavor_config, seed_manager, tokenizer_provider)
        self._prompt_rng = seed_manager.random("request_log_prompts")

        # Auto-assign session_id if missing (each row = one session)
        if "session_id" not in self.trace_df.columns:
            self.trace_df["session_id"] = range(len(self.trace_df))

        logger.info("Request log trace loaded: %d requests", len(self.trace_df))

    @property
    def required_columns(self) -> List[str]:
        return ["input_length", "output_length"]

    def prepare_session(self, group: pd.DataFrame) -> Session:
        """Create a single-request session from each trace group."""
        session_id = self._next_session_id()
        requests = {}

        for i, (_, row_series) in enumerate(group.iterrows()):
            row = row_series.to_dict()
            input_length = int(row["input_length"])
            output_length = int(row["output_length"])

            # Generate a random prompt of the target length
            if self.tokenizer.get_vocab is None:
                raise ValueError(
                    "Tokenizer must support get_vocab for request_log flavor"
                )
            vocab = self.tokenizer.get_vocab()
            vocab_size = len(vocab)
            token_ids = [
                self._prompt_rng.randint(0, vocab_size - 1) for _ in range(input_length)
            ]
            prompt_text = self.tokenizer.decode(token_ids)

            request = self._create_text_request(
                node_id=i,
                prompt_text=prompt_text,
                target_output_tokens=output_length,
                wait_after_ready=0.0,
                parent_node=i - 1 if i > 0 else None,
                target_prompt_tokens=input_length,
            )
            requests[i] = request

        session_graph = self._build_linear_session_graph(
            len(requests), [0.0] * len(requests)
        )

        return Session(
            id=session_id,
            session_graph=session_graph,
            requests=requests,
        )

    def wrap(self) -> pd.DataFrame:
        """Wrap trace for new epoch — just shuffle."""
        df = self.trace_df.copy()
        df["session_id"] = df["session_id"] + df["session_id"].max() + 1
        return self._shuffle_sessions(df)
