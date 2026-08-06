from __future__ import annotations

import pandas as pd
import pytest

from veeksha.config.generator.session import SeedTTSTextTraceFlavorConfig
from veeksha.generator.session.trace.seed_tts_text import (
    SeedTTSTextTraceFlavorGenerator,
)


def _generator(config: SeedTTSTextTraceFlavorConfig):
    generator = SeedTTSTextTraceFlavorGenerator.__new__(SeedTTSTextTraceFlavorGenerator)
    generator.flavor_config = config
    return generator


@pytest.mark.unit
def test_preserve_text_keeps_every_nonempty_prompt_and_slice_metadata() -> None:
    config = SeedTTSTextTraceFlavorConfig(
        dataset_name="example/dataset",
        preserve_text=True,
        min_tokens=20,
        max_tokens=20,
        metadata_columns=["language", "category"],
    )
    raw = pd.DataFrame(
        [
            {"text": "  नमस्ते दुनिया  ", "language": "hi", "category": "names"},
            {"text": "hello", "language": "en", "category": "short"},
        ]
    )

    prepared = _generator(config)._prepare_trace_df(raw)

    assert prepared["text"].tolist() == ["  नमस्ते दुनिया  ", "hello"]
    assert prepared["language"].tolist() == ["hi", "en"]
    assert prepared["category"].tolist() == ["names", "short"]


@pytest.mark.unit
def test_expected_rows_accepts_exact_canonical_selection() -> None:
    config = SeedTTSTextTraceFlavorConfig(
        dataset_name="example/dataset",
        preserve_text=True,
        expected_rows=2,
    )

    prepared = _generator(config)._prepare_trace_df(
        pd.DataFrame([{"text": "one"}, {"text": "two"}])
    )

    assert prepared["text"].tolist() == ["one", "two"]


@pytest.mark.unit
def test_expected_rows_rejects_wrong_source_selection() -> None:
    config = SeedTTSTextTraceFlavorConfig(
        dataset_name="example/dataset",
        preserve_text=True,
        expected_rows=2,
    )

    with pytest.raises(ValueError, match="selected 1 source rows; expected exactly 2"):
        _generator(config)._prepare_trace_df(pd.DataFrame([{"text": "one"}]))


@pytest.mark.unit
def test_expected_rows_rejects_prompts_skipped_during_preparation() -> None:
    config = SeedTTSTextTraceFlavorConfig(
        dataset_name="example/dataset",
        preserve_text=True,
        expected_rows=2,
    )

    with pytest.raises(
        ValueError,
        match=(
            "prepared 1 prompts from 2 source rows; expected exactly 2.*"
            "Skipped empty prompts: 1"
        ),
    ):
        _generator(config)._prepare_trace_df(
            pd.DataFrame([{"text": "one"}, {"text": "   "}])
        )


@pytest.mark.unit
def test_expected_rows_rejects_length_filtered_prompts() -> None:
    config = SeedTTSTextTraceFlavorConfig(
        dataset_name="example/dataset",
        min_tokens=2,
        max_tokens=2,
        expected_rows=2,
    )

    with pytest.raises(
        ValueError,
        match=(
            "prepared 1 prompts from 2 source rows; expected exactly 2.*"
            "skipped short prompts: 1"
        ),
    ):
        _generator(config)._prepare_trace_df(
            pd.DataFrame([{"text": "one two"}, {"text": "one"}])
        )


@pytest.mark.unit
def test_expected_rows_is_disabled_by_default() -> None:
    config = SeedTTSTextTraceFlavorConfig(
        dataset_name="example/dataset",
        preserve_text=True,
    )

    prepared = _generator(config)._prepare_trace_df(
        pd.DataFrame([{"text": "one"}, {"text": "   "}])
    )

    assert prepared["text"].tolist() == ["one"]


@pytest.mark.unit
@pytest.mark.parametrize("expected_rows", [0, -1])
def test_expected_rows_must_be_positive(expected_rows: int) -> None:
    with pytest.raises(ValueError, match="expected_rows must be positive"):
        SeedTTSTextTraceFlavorConfig(
            dataset_name="example/dataset",
            expected_rows=expected_rows,
        )


@pytest.mark.unit
def test_metadata_columns_must_be_scalar() -> None:
    config = SeedTTSTextTraceFlavorConfig(
        dataset_name="example/dataset",
        preserve_text=True,
        metadata_columns=["language"],
    )

    with pytest.raises(TypeError, match="must contain scalar values"):
        _generator(config)._prepare_trace_df(
            pd.DataFrame([{"text": "hello", "language": ["en"]}])
        )
