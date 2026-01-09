from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("lm_eval")

from veeksha.generator.session.lmeval import LMEvalSessionGenerator
from veeksha.types import ChannelModality
from veeksha.types import LMEvalOutputType


@dataclass
class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        # Deterministic, minimal tokenizer for unit tests (not model-accurate).
        return list(range(len(text.split())))


@dataclass
class _FakeInstance:
    request_type: str
    args: tuple[Any, ...]


def test_lmeval_loglikelihood_uses_completions_max_tokens_param() -> None:
    gen = LMEvalSessionGenerator.__new__(LMEvalSessionGenerator)
    gen.text_tokenizer = _FakeTokenizer()
    gen.ctxlen_tokens_by_request_id = {}

    inst = _FakeInstance(
        request_type=str(LMEvalOutputType.LOGLIKELIHOOD),
        args=("Question: X\nAnswer:", " Y"),
    )
    req = LMEvalSessionGenerator._build_request(gen, request_id=0, instance=inst)  # type: ignore[arg-type]

    sp = req.metadata["sampling_params"]
    assert sp["stream"] is False
    assert sp["echo"] is True
    assert sp["logprobs"] == 1
    assert sp["temperature"] == 0
    assert sp["max_tokens"] == 1
    assert "max_completion_tokens" not in sp

    assert req.metadata["api_mode"] == "completions"
    assert ChannelModality.TEXT in req.channels


