from __future__ import annotations

import math

import pytest

from veeksha.evaluator.accuracy.base import LMEvalAccuracyEvaluator


def _parse(ctxlen: int, lp: dict) -> tuple[float, bool]:
    # Instantiate without calling __init__ to avoid heavyweight config wiring.
    ev = LMEvalAccuracyEvaluator.__new__(LMEvalAccuracyEvaluator)
    return LMEvalAccuracyEvaluator._parse_logprobs(ev, ctxlen=ctxlen, lp=lp)


def test_parse_logprobs_openai_dict_sums_target_and_ignores_last_completion() -> None:
    # token_logprobs includes ctx (2 toks), target (2 toks), completion (1 tok).
    lp = {
        "token_logprobs": [-10.0, -9.0, -1.5, -2.5, -99.0],
        "top_logprobs": [
            {"a": -10.0},
            {"b": -9.0},
            {"T1": -1.5, "x": -2.0},
            {"T2": -2.5, "y": -3.0},
            {"C": -99.0, "z": -100.0},
        ],
    }
    total, is_greedy = _parse(ctxlen=2, lp=lp)
    assert total == pytest.approx(-4.0)
    assert is_greedy is True


def test_parse_logprobs_openai_dict_marks_non_greedy_when_top_beats_token() -> None:
    lp = {
        "token_logprobs": [-1.0, -2.0, -3.0],
        "top_logprobs": [
            {"a": -1.0},
            {"b": -1.9, "other": -1.0},  # token is not argmax
            {"c": -3.0},
        ],
    }
    total, is_greedy = _parse(ctxlen=0, lp=lp)
    # Slicing ctxlen:-1 ignores the last entry, so sum is over first two.
    assert total == pytest.approx(-3.0)
    assert is_greedy is False


def test_parse_logprobs_content_list() -> None:
    lp = {
        "content": [
            {"token": "c1", "logprob": -9.0, "top_logprobs": [{"token": "c1", "logprob": -9.0}]},
            {"token": "c2", "logprob": -8.0, "top_logprobs": [{"token": "c2", "logprob": -8.0}]},
            {"token": "t1", "logprob": -1.0, "top_logprobs": [{"token": "t1", "logprob": -1.0}]},
            {"token": "t2", "logprob": -2.0, "top_logprobs": [{"token": "t2", "logprob": -2.0}]},
        ]
    }
    total, is_greedy = _parse(ctxlen=2, lp=lp)
    assert total == pytest.approx(-3.0)
    assert is_greedy is True


def test_parse_logprobs_chunks_list_token_logprobs_fallback() -> None:
    lp = {
        "chunks": [
            {"token_logprobs": [-1.0], "top_logprobs": [{"token": "a", "logprob": -1.0}]},
            {"token_logprobs": [-2.0], "top_logprobs": [{"token": "b", "logprob": -2.0}]},
        ]
    }
    total, is_greedy = _parse(ctxlen=123, lp=lp)
    assert total == pytest.approx(-3.0)
    assert is_greedy is True


def test_parse_logprobs_raises_on_unknown_format() -> None:
    with pytest.raises(KeyError):
        _parse(ctxlen=0, lp={"weird": 1})


