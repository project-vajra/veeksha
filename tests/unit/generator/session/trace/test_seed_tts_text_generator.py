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
    """Minimal stand-in for an HF dataset exposing to_pandas()."""

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
        seed_tts_text, "_load_hf_dataset", lambda cfg: _FakeDataset(frame)
    )

    config = TraceSessionGeneratorConfig(
        trace_file="", flavor=flavor_config, wrap_mode=wrap_mode
    )
    tokenizer_provider = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda t: t.split()
    tokenizer_provider.for_modality.return_value = tokenizer

    return SeedTTSTextTraceFlavorGenerator(
        config, flavor_config, SeedManager(seed=seed), tokenizer_provider
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_target_duration_rejects_char_based_length() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        SeedTTSTextTraceFlavorConfig(
            target_duration_s=180.0, min_chars=10, max_chars=20
        )


@pytest.mark.unit
def test_target_duration_spread_requires_target() -> None:
    with pytest.raises(ValueError, match="spread_s requires target_duration_s"):
        SeedTTSTextTraceFlavorConfig(target_duration_spread_s=30.0)


@pytest.mark.unit
def test_target_duration_rejects_nonpositive_words_per_second() -> None:
    with pytest.raises(ValueError, match="words_per_second must be positive"):
        SeedTTSTextTraceFlavorConfig(target_duration_s=180.0, words_per_second=0.0)


# ---------------------------------------------------------------------------
# Generator: long target honored
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_long_target_duration_maps_to_long_word_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 240s * 2.5 words/s == 600 words: a multi-minute soak session.
    flavor_config = SeedTTSTextTraceFlavorConfig(
        local_path="dummy", target_duration_s=240.0, words_per_second=2.5
    )
    generator = _make_generator(monkeypatch, flavor_config, n_words=800)

    session = generator.generate_session()
    request = session.requests[0]

    assert request.metadata["input_words"] == 600
    assert request.metadata["target_duration_s"] == pytest.approx(240.0)


@pytest.mark.unit
def test_target_duration_spread_stays_within_bounds_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def word_counts(seed: int) -> list[int]:
        flavor_config = SeedTTSTextTraceFlavorConfig(
            local_path="dummy",
            target_duration_s=180.0,
            target_duration_spread_s=60.0,
            words_per_second=2.5,
        )
        generator = _make_generator(
            monkeypatch, flavor_config, n_words=2000, seed=seed, wrap_mode=True
        )
        counts = []
        for _ in range(20):
            session = generator.generate_session()
            counts.append(session.requests[0].metadata["input_words"])
        return counts

    counts = word_counts(7)
    # duration in [120, 240] s -> words in [300, 600] at 2.5 words/s.
    assert all(300 <= c <= 600 for c in counts)
    assert not all(c == counts[0] for c in counts)  # the spread actually varies
    # Same seed reproduces the same draws.
    assert word_counts(7) == counts


@pytest.mark.unit
def test_no_target_duration_uses_word_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flavor_config = SeedTTSTextTraceFlavorConfig(
        local_path="dummy", min_tokens=20, max_tokens=40
    )
    generator = _make_generator(monkeypatch, flavor_config, n_words=200)

    session = generator.generate_session()
    request = session.requests[0]

    assert 20 <= request.metadata["input_words"] <= 40
    assert "target_duration_s" not in request.metadata
