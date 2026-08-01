"""The preflight synthetic text prompt must stream as many paced deltas.

Regression guard: the preflight tokenizer decodes each token to "<int> " so
PromptStringGenerator tiles whitespace-separated words. Without the boundary
space the prompt collapses to a single space-less blob and segment_text yields
one delta -- which silently disables input-pacing/delivery measurement for the
streaming-text WS client (streaming_tts).
"""

import pytest

from veeksha.client.utils import segment_text
from veeksha.config.generator.channel import TextChannelGeneratorConfig
from veeksha.config.generator.length import FixedLengthGeneratorConfig
from veeksha.core.seeding import SeedManager
from veeksha.generator.channel.text import TextChannelGenerator
from veeksha.preflight.drivers import _build_preflight_tokenizer_provider
from veeksha.types import ChannelModality


def _handle():
    return _build_preflight_tokenizer_provider("preflight-mock").for_modality(
        ChannelModality.TEXT
    )


def _prompt(input_tokens: int) -> str:
    cfg = TextChannelGeneratorConfig(
        body_length_generator=FixedLengthGeneratorConfig(value=input_tokens)
    )
    gen = TextChannelGenerator(cfg, SeedManager(0), _handle())
    return gen.generate_content(is_root=True).input_text


@pytest.mark.unit
def test_decode_gives_per_token_boundary_space():
    handle = _handle()
    assert handle.decode([914, 612, 84]) == "914 612 84 "
    # The generator's tiling ("".join) must preserve word boundaries.
    assert handle.decode([1]) + handle.decode([2]) == "1 2 "


@pytest.mark.unit
@pytest.mark.parametrize("ids", [[5], [5, 5], [914, 612, 84], [0, 7, 42]])
def test_encode_decode_round_trips(ids):
    handle = _handle()
    assert list(handle.encode(handle.decode(ids))) == ids


@pytest.mark.unit
@pytest.mark.parametrize("input_tokens", [16, 100, 512])
def test_prompt_is_whitespace_delimited_and_multi_segment(input_tokens):
    text = _prompt(input_tokens)
    # Token count is honored (each token is one whitespace word).
    assert len(text.split()) == input_tokens
    # And it streams as ~input_tokens/tokens_per_delta deltas, not one blob.
    segments = segment_text(text, tokens_per_delta=4)
    assert len(segments) == pytest.approx(input_tokens / 4, abs=1)
    assert len(segments) > 1
    # segment_text's byte-exact round-trip invariant still holds.
    assert "".join(s.text for s in segments) == text
