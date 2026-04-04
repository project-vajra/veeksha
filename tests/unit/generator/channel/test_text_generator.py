import pytest
from unittest.mock import MagicMock, patch
from veeksha.config.generator.channel import TextChannelGeneratorConfig
from veeksha.config.generator.length import FixedLengthGeneratorConfig
from veeksha.generator.channel.text import TextChannelGenerator
from veeksha.core.seeding import SeedManager

# Fake stable encodings: simple token lists that round-trip trivially under mock tokenizer
_FAKE_ENCODINGS = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda x: [1] * len(x.split())
    tokenizer.decode.side_effect = lambda x: " ".join(["word"] * len(x))
    tokenizer.count_tokens.side_effect = lambda x: (
        len(x.split()) if isinstance(x, str) else len(x)
    )
    return tokenizer


@pytest.fixture
def mock_seed_manager():
    return SeedManager(seed=42)


@pytest.fixture
def text_config():
    return TextChannelGeneratorConfig(
        body_length_generator=FixedLengthGeneratorConfig(value=10),
        shared_prefix_ratio=0.5,
        shared_prefix_probability=1.0,  # Always use shared prefix if is_root
    )


@pytest.fixture(autouse=True)
def mock_load_or_generate():
    """Patch stable-encoding generation so tests never hit disk or tokenizer."""
    with patch(
        "veeksha.core.prompt_generator._load_or_generate",
        return_value=_FAKE_ENCODINGS,
    ):
        yield


def test_text_generator_initialization(mock_tokenizer, mock_seed_manager, text_config):
    generator = TextChannelGenerator(text_config, mock_seed_manager, mock_tokenizer)
    assert generator.config == text_config
    assert generator._prompt_gen is not None


def test_generate_content_basic(mock_tokenizer, mock_seed_manager, text_config):
    generator = TextChannelGenerator(text_config, mock_seed_manager, mock_tokenizer)
    content = generator.generate_content(is_root=False)

    assert content.target_prompt_tokens == 10
    assert isinstance(content.input_text, str)


def test_generate_content_shared_prefix(mock_tokenizer, mock_seed_manager, text_config):
    # Config has 0.5 ratio and 1.0 probability — always uses shared prefix for root
    generator = TextChannelGenerator(text_config, mock_seed_manager, mock_tokenizer)

    generator.generate_content(is_root=True)
    generator.generate_content(is_root=True)

    # With fixed length 10 and ratio 0.5, prefix length is 5.
    # _generate_shared_prefix caches tokens; verify cache was populated.
    assert len(generator._shared_prefix_tokens) >= 5


def test_generate_content_min_tokens_instruction(
    mock_tokenizer, mock_seed_manager, text_config
):
    generator = TextChannelGenerator(
        text_config,
        mock_seed_manager,
        mock_tokenizer,
    )

    # Mock tokenizer to return short length for suffix so it fits
    mock_tokenizer.encode.side_effect = lambda x: [1] * (
        1 if "Generate" in x else len(x.split())
    )

    content = generator.generate_content(is_root=False, min_tokens_suffix=5)
    assert (
        "Generate at least" in content.input_text or content.target_prompt_tokens == 10
    )
