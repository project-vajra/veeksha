"""Timed synthetic session trace flavor generator."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd  # type: ignore[import]

from veeksha.config.generator.session import (
    TimedSyntheticSessionTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import topological_order
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.generator.session.trace.base_flavor import (
    TraceFlavorGeneratorBase,
    TraceNodeContext,
)
from veeksha.generator.session.trace.prompt_builder import TracePromptBuilder
from veeksha.logger import init_logger

logger = init_logger(__name__)


class TimedSyntheticSessionTraceFlavorGenerator(TraceFlavorGeneratorBase):
    """Timed synthetic session trace flavor generator.

    Replays timed session traces with synthetic prompt generation.
    Session topology is read from ``session_context`` when present,
    matching Veeksha's dispatch trace format. Legacy traces without
    ``session_context`` fall back to linear row-order topology.
    """

    _PROMPT_COL = "_tss_prompt_text"
    _SEED_COL = "_tss_lineage_seed"

    def __init__(
        self,
        config: TraceSessionGeneratorConfig,
        flavor_config: TimedSyntheticSessionTraceFlavorConfig,
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
        self._lineage_seed_rng = seed_manager.random("tss_lineage_seeds")

        self.trace_df = self._normalize_trace_topology(self.trace_df)
        logger.info("Materializing prompts for timed synthetic session trace...")
        self.trace_df = self._materialize_prompts(self.trace_df)

    @property
    def required_columns(self) -> List[str]:
        return [
            "session_id",
            "input_length",
            "new_input_length",
            "output_length",
        ]

    def prepare_session(self, group: pd.DataFrame) -> Session:
        """Prepare a DAG session with lineage-aware synthetic prompts."""
        session_id = self._next_session_id()
        rows_by_node, node_contexts = self._group_rows_and_contexts(group)
        session_graph = self._build_session_graph_from_contexts(node_contexts)
        contexts_by_node = {ctx.node_id: ctx for ctx in node_contexts}
        requests = {}

        for node_id in topological_order(session_graph):
            row = rows_by_node[node_id]
            ctx = contexts_by_node[node_id]
            prompt_tokens = cast(int, row["new_input_length"])
            output_length = cast(int, row["output_length"])
            prompt_text = row.get(self._PROMPT_COL)
            if prompt_text is None:
                raise ValueError(
                    f"Prompt cache missing for session {session_id} node {node_id}."
                )

            request = self._create_text_request(
                node_id=node_id,
                prompt_text=cast(str, prompt_text),
                target_output_tokens=output_length,
                wait_after_ready=ctx.wait_after_ready,
                parent_nodes=ctx.parent_nodes,
                history_parent=ctx.history_parent,
                target_prompt_tokens=prompt_tokens,
            )
            requests[node_id] = request

        return Session(
            id=session_id,
            session_graph=session_graph,
            requests=requests,
        )

    def wrap(self) -> pd.DataFrame:
        """Wrap trace for new epoch with fresh lineage seeds."""
        df = self.trace_df.copy()
        if not df.empty:
            df["session_id"] = df["session_id"] + cast(int, df["session_id"].max()) + 1
        df = self._shuffle_sessions(df)
        return self._materialize_prompts(df)

    def _normalize_trace_topology(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize per-row ``session_context`` to a canonical replay contract."""
        df = df.copy()
        if "session_context" not in df.columns:
            df["session_context"] = None

        for _, group in df.groupby("session_id", sort=False):
            present_contexts: List[tuple[int, TraceNodeContext]] = []
            missing_context_rows: List[int] = []

            for idx, row in group.iterrows():
                raw = row.get("session_context")
                if self._is_missing_session_context(raw):
                    missing_context_rows.append(idx)
                    continue
                present_contexts.append((idx, self._parse_session_context(raw)))

            if present_contexts and missing_context_rows:
                raise ValueError(
                    "Mixed topology metadata in one session. Either every row must "
                    "provide session_context or none of them may."
                )

            if not present_contexts:
                inferred = self._infer_linear_session_contexts(group)
                for idx, ctx in inferred.items():
                    df.at[idx, "session_context"] = ctx
                continue

            self._build_session_graph_from_contexts(
                [ctx for _, ctx in present_contexts]
            )
            for idx, ctx in present_contexts:
                df.at[idx, "session_context"] = self._create_session_context(
                    node_id=ctx.node_id,
                    wait_after_ready=ctx.wait_after_ready,
                    parent_nodes=ctx.parent_nodes,
                    history_parent=ctx.history_parent,
                )

        return df

    def _materialize_prompts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate prompts with lineage seeds derived from history parents."""
        df = df.drop(columns=[self._PROMPT_COL, self._SEED_COL], errors="ignore").copy()
        df[self._SEED_COL] = None
        df[self._PROMPT_COL] = None

        for _, group in df.groupby("session_id", sort=False):
            rows_by_node, node_contexts = self._group_rows_and_contexts(group)
            session_graph = self._build_session_graph_from_contexts(node_contexts)
            contexts_by_node = {ctx.node_id: ctx for ctx in node_contexts}
            lineage_seeds: Dict[int, int] = {}

            for node_id in topological_order(session_graph):
                ctx = contexts_by_node[node_id]
                if ctx.history_parent is None:
                    lineage_seeds[node_id] = self._lineage_seed_rng.getrandbits(32)
                else:
                    lineage_seeds[node_id] = lineage_seeds[ctx.history_parent]

                row = rows_by_node[node_id]
                prompt_tokens = cast(int, row["new_input_length"])
                prompt = self.prompt_builder.generate_unique_prompt(
                    num_tokens=prompt_tokens,
                    page_size=self.flavor_config.page_size,
                    seed=lineage_seeds[node_id],
                )
                df.at[row.name, self._SEED_COL] = lineage_seeds[node_id]
                df.at[row.name, self._PROMPT_COL] = prompt

        return df

    def _group_rows_and_contexts(
        self, group: pd.DataFrame
    ) -> tuple[Dict[int, pd.Series], List[TraceNodeContext]]:
        rows_by_node: Dict[int, pd.Series] = {}
        contexts: List[TraceNodeContext] = []

        for _, row in group.iterrows():
            ctx = self._parse_session_context(row["session_context"])
            if ctx.node_id in rows_by_node:
                raise ValueError(
                    f"Duplicate node_id {ctx.node_id} found in session "
                    f"{row['session_id']}."
                )
            rows_by_node[ctx.node_id] = row
            contexts.append(ctx)

        return rows_by_node, contexts

    def _infer_linear_session_contexts(
        self, group: pd.DataFrame
    ) -> Dict[int, Dict[str, object]]:
        """Backfill legacy linear traces that predate ``session_context``."""
        contexts: Dict[int, Dict[str, object]] = {}
        ordered = group
        use_turn_idx = "turn_idx" in group.columns
        if use_turn_idx:
            ordered = group.sort_values("turn_idx", kind="stable")

        ordered_node_ids: List[int] = []
        for pos, (_, row) in enumerate(ordered.iterrows()):
            node_id = cast(int, row["turn_idx"]) if use_turn_idx else pos
            ordered_node_ids.append(node_id)

        for pos, (idx, row) in enumerate(ordered.iterrows()):
            node_id = ordered_node_ids[pos]
            parent_nodes = [ordered_node_ids[pos - 1]] if pos > 0 else []
            history_parent = parent_nodes[0] if parent_nodes else None
            contexts[idx] = self._create_session_context(
                node_id=node_id,
                wait_after_ready=self._coerce_wait_after_ready(
                    row.get("wait_after_previous_response_s")
                ),
                parent_nodes=parent_nodes,
                history_parent=history_parent,
            )

        return contexts

    @staticmethod
    def _is_missing_session_context(raw: Any) -> bool:
        if raw is None:
            return True
        if isinstance(raw, str):
            return raw.strip() == "" or raw.strip().lower() == "null"
        try:
            return bool(pd.isna(raw))
        except TypeError:
            return False

    def _parse_session_context(self, raw: Any) -> TraceNodeContext:
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError(
                "session_context must be a dict or JSON object string, got "
                f"{type(raw).__name__}."
            )

        required_keys = {"node_id", "parent_nodes", "history_parent", "wait_after_ready"}
        missing_keys = required_keys - set(raw.keys())
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"session_context is missing required keys: {missing}")

        parent_nodes_raw = raw["parent_nodes"]
        if parent_nodes_raw is None:
            parent_nodes: List[int] = []
        elif isinstance(parent_nodes_raw, list):
            parent_nodes = [int(node_id) for node_id in parent_nodes_raw]
        else:
            raise ValueError("session_context.parent_nodes must be a list or null.")

        history_parent_raw = raw["history_parent"]
        history_parent = None
        if history_parent_raw is not None:
            history_parent = int(history_parent_raw)

        return TraceNodeContext(
            node_id=int(raw["node_id"]),
            parent_nodes=parent_nodes,
            history_parent=history_parent,
            wait_after_ready=self._coerce_wait_after_ready(raw["wait_after_ready"]),
        )

    @staticmethod
    def _coerce_wait_after_ready(raw: Any) -> float:
        if raw is None:
            return 0.0
        try:
            if pd.isna(raw):
                return 0.0
        except TypeError:
            pass
        return float(raw)
