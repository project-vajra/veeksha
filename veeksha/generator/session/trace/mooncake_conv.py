"""Mooncake Conversation trace flavor generator."""

import random
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from veeksha.config.generator.session import (
    MooncakeConvTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.tokenizer import TokenizerProvider, gen_prompt_from_corpus
from veeksha.generator.session.trace.base_flavor import TraceFlavorGeneratorBase
from veeksha.generator.session.trace.prompt_builder import TracePromptBuilder


def _build_epoch_hash_id_map(
    unique_list: List[int], rng: random.Random
) -> Dict[int, int]:
    """Build a mapping from old hash IDs to new unique hash IDs."""
    used: Dict[int, bool] = {}
    id_map: Dict[int, int] = {}
    for src in unique_list:
        dst = rng.getrandbits(32)
        _i = 0
        while dst == 0 or dst in used:
            dst = rng.getrandbits(32)
            _i += 1
            if _i > 1000:
                raise ValueError(
                    f"Could not generate a non-colliding positive remapped ID for {src}"
                )
        id_map[src] = int(dst)
        used[dst] = True
    return id_map


def _remap_trace_hash_ids(
    trace_df: pd.DataFrame, rng: random.Random, keep_fixed: Optional[Set[int]] = None
) -> pd.DataFrame:
    """Remap hash IDs in trace, optionally keeping some fixed."""
    df = trace_df.copy()
    keep_fixed = keep_fixed or set()

    unique_ids: Set[int] = set()
    for ids in trace_df["hash_ids"]:
        unique_ids.update(ids)
    unique_ids = unique_ids - keep_fixed
    unique_list = sorted(unique_ids)

    if unique_list:
        id_map = _build_epoch_hash_id_map(unique_list, rng)
        for id_ in keep_fixed:
            id_map[id_] = id_
        df["hash_ids"] = trace_df["hash_ids"].apply(
            lambda lst: [id_map.get(x, x) for x in lst]
        )
    return df


class MooncakeConvTraceFlavorGenerator(TraceFlavorGeneratorBase):
    """Mooncake Conversation trace flavor generator.

    Supports traces with explicit hash_id-based content sharing across sessions.
    Same hash_id produces identical prompt content, modeling prefix sharing.
    """

    def __init__(
        self,
        config: TraceSessionGeneratorConfig,
        flavor_config: MooncakeConvTraceFlavorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
    ):
        super().__init__(config, flavor_config, seed_manager, tokenizer_provider)
        self.flavor_config = flavor_config
        self.tokenizer_provider = tokenizer_provider

        # Initialize prompt builder with corpus
        self.prompt_builder = TracePromptBuilder(
            tokenizer=self.tokenizer,
            seed_manager=seed_manager.child("prompt_builder"),
            corpus_file=(
                Path(flavor_config.corpus_file) if flavor_config.corpus_file else None
            ),
        )
        self._wrap_rng = seed_manager.random("mooncake_conv_wrapping")
        self._corpus_rng = seed_manager.random("mooncake_conv_corpus")

    @property
    def required_columns(self) -> List[str]:
        return [
            "session_id",
            "input_length",
            "output_length",
            "new_input_length",
            "hash_ids",
        ]

    def prepare_session(self, group: pd.DataFrame) -> Session:
        """Prepare session using hash_id-based prompt generation."""
        session_id = self._next_session_id()
        requests = {}
        wait_times: List[float] = []

        for i, (_, row) in enumerate(group.iterrows()):
            if i == 0:
                # first request uses hash_ids for deterministic sharing
                hash_ids_list: List[int] = list(row["hash_ids"])

                prompt_text = self.prompt_builder.build_from_hash_ids(
                    hash_ids=hash_ids_list,
                    block_size=self.flavor_config.block_size,
                )

                calculated_len = self.flavor_config.block_size * len(hash_ids_list)

            else:
                # subsequent requests: random generation from corpus
                length = int(row["new_input_length"])

                prompt_text = gen_prompt_from_corpus(
                    num_tokens=length,
                    pretokenized_lines=self.prompt_builder.pretokenized_lines,
                    tokenizer_handle=self.tokenizer,
                    rng=self._corpus_rng,
                )

                calculated_len = int(row["new_input_length"])

            # wait time
            wait_time_val = row.get("wait_after_previous_response_s")
            if wait_time_val is None or pd.isna(wait_time_val):
                wait_time = 0.0
            else:
                wait_time = float(wait_time_val)
            wait_times.append(wait_time)

            # output length
            output_length = int(row["output_length"])

            request = self._create_text_request(
                node_id=i,
                prompt_text=prompt_text,
                target_output_tokens=output_length,
                wait_after_ready=wait_time,
                parent_node=i - 1 if i > 0 else None,
                target_prompt_tokens=calculated_len,
            )
            requests[i] = request

        session_graph = self._build_linear_session_graph(len(requests), wait_times)

        return Session(
            id=session_id,
            session_graph=session_graph,
            requests=requests,
        )

    def wrap(self) -> pd.DataFrame:
        """Wrap trace for new epoch.

        The mooncake conversation trace has sharing across sessions.
        Only the first hash_id is shared across all sessions, so we
        keep it fixed and remap the rest.
        """
        df = self.trace_df
        first_hash_id = df.iloc[0]["hash_ids"][0]

        # keeps first hash id fixed
        df = _remap_trace_hash_ids(df, self._wrap_rng, keep_fixed={first_hash_id})

        # increment session IDs
        df["session_id"] = df["session_id"] + df["session_id"].max() + 1

        return self._shuffle_sessions(df)
