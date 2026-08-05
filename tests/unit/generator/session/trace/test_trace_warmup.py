from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd

from veeksha.config.generator.session import TraceSessionGeneratorConfig
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.generator.session.trace.base_flavor import TraceFlavorGeneratorBase


class _WarmupTraceFlavor(TraceFlavorGeneratorBase):
    @property
    def required_columns(self):
        return ["session_id", "prompt"]

    def prepare_session(self, group: pd.DataFrame) -> Session:
        return Session(
            id=int(group["session_id"].iloc[0]),
            session_graph=MagicMock(),
            requests={},
        )

    def wrap(self) -> pd.DataFrame:
        return self.trace_df


def test_configured_warmup_replays_prefix_without_consuming_measured_trace(
    tmp_path,
) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(
            json.dumps({"session_id": index, "prompt": f"prompt-{index}"}) + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    flavor = MagicMock()
    tokenizer_provider = MagicMock()
    generator = _WarmupTraceFlavor(
        TraceSessionGeneratorConfig(
            trace_file=str(trace),
            flavor=flavor,
            wrap_mode=False,
            warmup_sessions=2,
        ),
        flavor,
        SeedManager(seed=42),
        tokenizer_provider,
    )

    assert [session.id for session in generator.get_warmup_sessions()] == [0, 1]
    assert generator.generate_session().id == 0
