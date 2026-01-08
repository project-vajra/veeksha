"""Context-Cached trace flavor generator."""

from pathlib import Path
from typing import List, Optional

import pandas as pd  # type: ignore[import]

from veeksha.config.generator.session import (
    ClaudeCodeTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.generator.session.trace.base_flavor import TraceFlavorGeneratorBase
from veeksha.generator.session.trace.prompt_builder import TracePromptBuilder
from veeksha.logger import init_logger

logger = init_logger(__name__)


class ClaudeCodeTraceFlavorGenerator(TraceFlavorGeneratorBase):
    """ClaudeCode trace flavor generator."""

    _PROMPT_COL = "_cc_prompt_text"
    _SEED_COL = "_cc_session_seed"

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

        logger.info("Materializing prompts for ClaudeCode trace...")
        self.trace_df = self._materialize_prompts(self.trace_df)

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

        for i, (_, row) in enumerate(group.iterrows()):
            prompt_tokens = int(row["new_input_length"])
            output_length = int(row["output_length"])

            prompt_text = row.get(self._PROMPT_COL)
            if prompt_text is None:
                raise ValueError("Prompt cache missing for ClaudeCode trace row.")

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
                target_prompt_tokens=prompt_tokens,
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
        df = self._shuffle_sessions(df)
        return self._materialize_prompts(df, first_turn_only=True)

    def _materialize_prompts(
        self, df: pd.DataFrame, *, first_turn_only: bool = False
    ) -> pd.DataFrame:
        """Generate prompts, optionally only for the first turn of each session."""

        if first_turn_only:
            df = df.copy()
        else:
            df = df.drop(
                columns=[self._PROMPT_COL, self._SEED_COL], errors="ignore"
            ).copy()

        if self._SEED_COL not in df.columns:
            df[self._SEED_COL] = None
        if self._PROMPT_COL not in df.columns:
            df[self._PROMPT_COL] = None

        mask = self._get_first_turn_mask(df)

        generated = 0
        session_seeds: dict[int, int] = {}
        for idx, row in df.iterrows():
            if first_turn_only and not mask.loc[idx]:
                continue

            session_id = int(row["session_id"])
            prompt_tokens = int(row["new_input_length"])
            seed: Optional[int] = session_seeds.get(session_id)
            if seed is None or (first_turn_only and mask.loc[idx]):
                existing = row[self._SEED_COL]
                if not first_turn_only and existing is not None:
                    seed = int(existing)
                else:
                    seed = self._session_seed_rng.getrandbits(32)
                session_seeds[session_id] = seed

            df.at[idx, self._SEED_COL] = seed

            prompt = self.prompt_builder.generate_unique_prompt(
                num_tokens=prompt_tokens,
                page_size=self.flavor_config.page_size,
                seed=seed,
            )
            df.at[idx, self._PROMPT_COL] = prompt
            generated += 1

        return df

    @staticmethod
    def _get_first_turn_mask(df: pd.DataFrame) -> pd.Series:
        """Return boolean mask indicating the first request of each session."""
        if "turn_idx" in df.columns:
            first_turns = df.groupby("session_id")["turn_idx"].transform("min")
            return df["turn_idx"] == first_turns

        ordered = df.reset_index()
        first_flags = ordered.groupby("session_id").cumcount() == 0
        mask = pd.Series(False, index=df.index)
        mask.loc[ordered["index"]] = first_flags.values
        return mask
