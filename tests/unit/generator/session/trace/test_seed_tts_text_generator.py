from unittest.mock import MagicMock

import pandas as pd
import pytest

import veeksha.generator.session.trace.seed_tts_text as seed_tts_text
from veeksha.config.generator.session import (
    SeedTTSTextTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.generator.session.trace.seed_tts_text import (
    SeedTTSTextTraceFlavorGenerator,
)


class _FakeDataset:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame


def _make_generator(
    monkeypatch: pytest.MonkeyPatch,
    flavor_config: SeedTTSTextTraceFlavorConfig,
    *,
    n_words: int,
    seed: int = 42,
    wrap_mode: bool = False,
) -> SeedTTSTextTraceFlavorGenerator:
    text = " ".join(f"word{i}" for i in range(n_words))
    frame = pd.DataFrame([{"text": text, "filename": "row0"}])
    monkeypatch.setattr(
        seed_tts_text,
        "_load_hf_dataset",
        lambda _config: _FakeDataset(frame),
    )

    config = TraceSessionGeneratorConfig(
        trace_file="",
        flavor=flavor_config,
        wrap_mode=wrap_mode,
    )
    tokenizer_provider = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text: text.split()
    tokenizer_provider.for_modality.return_value = tokenizer

    return SeedTTSTextTraceFlavorGenerator(
        config,
        flavor_config,
        SeedManager(seed=seed),
        tokenizer_provider,
    )


@pytest.mark.unit
def test_target_duration_rejects_char_based_length() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        SeedTTSTextTraceFlavorConfig(
            target_duration_s=180.0,
            min_chars=10,
            max_chars=20,
        )


@pytest.mark.unit
def test_target_duration_spread_requires_target() -> None:
    with pytest.raises(ValueError, match="spread_s requires target_duration_s"):
        SeedTTSTextTraceFlavorConfig(target_duration_spread_s=30.0)


@pytest.mark.unit
def test_target_duration_rejects_nonpositive_words_per_second() -> None:
    with pytest.raises(ValueError, match="words_per_second must be positive"):
        SeedTTSTextTraceFlavorConfig(
            target_duration_s=180.0,
            words_per_second=0.0,
        )


@pytest.mark.unit
def test_long_target_duration_maps_to_long_word_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flavor_config = SeedTTSTextTraceFlavorConfig(
        local_path="dummy",
        target_duration_s=240.0,
        words_per_second=2.5,
    )
    generator = _make_generator(monkeypatch, flavor_config, n_words=800)

    request = generator.generate_session().requests[0]

    assert request.metadata["input_words"] == 600
    assert request.metadata["target_duration_s"] == pytest.approx(240.0)


@pytest.mark.unit
def test_target_duration_spread_is_bounded_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def samples(seed: int) -> list[tuple[int, float]]:
        flavor_config = SeedTTSTextTraceFlavorConfig(
            local_path="dummy",
            target_duration_s=180.0,
            target_duration_spread_s=60.0,
            words_per_second=2.5,
        )
        generator = _make_generator(
            monkeypatch,
            flavor_config,
            n_words=2000,
            seed=seed,
            wrap_mode=True,
        )
        sampled = []
        for _ in range(20):
            request = generator.generate_session().requests[0]
            sampled.append(
                (
                    request.metadata["input_words"],
                    request.metadata["target_duration_s"],
                )
            )
        return sampled

    sampled = samples(7)

    assert all(300 <= word_count <= 600 for word_count, _ in sampled)
    assert all(120.0 <= duration_s <= 240.0 for _, duration_s in sampled)
    assert len(set(sampled)) > 1
    assert samples(7) == sampled


@pytest.mark.unit
def test_target_duration_rejects_rows_shorter_than_largest_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flavor_config = SeedTTSTextTraceFlavorConfig(
        local_path="dummy",
        target_duration_s=10.0,
        target_duration_spread_s=2.0,
        words_per_second=2.0,
    )

    with pytest.raises(ValueError, match="at least 24 words"):
        _make_generator(monkeypatch, flavor_config, n_words=23)


@pytest.mark.unit
def test_no_target_duration_uses_word_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flavor_config = SeedTTSTextTraceFlavorConfig(
        local_path="dummy",
        min_tokens=20,
        max_tokens=40,
    )
    generator = _make_generator(monkeypatch, flavor_config, n_words=200)

    request = generator.generate_session().requests[0]

    assert 20 <= request.metadata["input_words"] <= 40
    assert "target_duration_s" not in request.metadata
