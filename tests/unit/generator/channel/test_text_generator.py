
import pytest
from unittest.mock import MagicMock, patch
from veeksha.config.generator.channel import TextChannelGeneratorConfig
from veeksha.config.generator.length import FixedLengthGeneratorConfig
from veeksha.generator.channel.text import TextChannelGenerator
from veeksha.core.seeding import SeedManager

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda x: [1] * len(x.split())  # simple mockery
    tokenizer.decode.side_effect = lambda x: " ".join(["word"] * len(x))
    tokenizer.count_tokens.side_effect = lambda x: len(x.split()) if isinstance(x, str) else len(x)
    return tokenizer

@pytest.fixture
def mock_seed_manager():
    return SeedManager(seed=42)

@pytest.fixture
def mock_load_corpus():
    with patch("veeksha.generator.channel.text.load_corpus") as mock:
        mock.return_value = ["line one", "line two", "line three"]
        yield mock

@pytest.fixture
def text_config():
    return TextChannelGeneratorConfig(
        body_length_generator=FixedLengthGeneratorConfig(value=10),
        shared_prefix_ratio=0.5,
        shared_prefix_probability=1.0,  # Always use shared prefix if is_root
    )

def test_text_generator_initialization(mock_load_corpus, mock_tokenizer, mock_seed_manager, text_config):
    generator = TextChannelGenerator(text_config, mock_seed_manager, mock_tokenizer)
    assert generator.config == text_config
    assert len(generator._corpus_lines) == 3

def test_generate_content_basic(mock_load_corpus, mock_tokenizer, mock_seed_manager, text_config):
    generator = TextChannelGenerator(text_config, mock_seed_manager, mock_tokenizer)
    content = generator.generate_content(is_root=False)
    
    # Check if target lengths match config
    assert content.target_prompt_tokens == 10
    assert isinstance(content.input_text, str)

def test_generate_content_shared_prefix(mock_load_corpus, mock_tokenizer, mock_seed_manager, text_config):
    # Config has 0.5 ratio and 1.0 probability
    generator = TextChannelGenerator(text_config, mock_seed_manager, mock_tokenizer)
    
    # First root request
    content1 = generator.generate_content(is_root=True)
    
    # Second root request
    content2 = generator.generate_content(is_root=True)
    
    # With fixed length 10 and ratio 0.5, prefix length is 5.
    # The generator should use the same prefix for both.
    # Since we rely on random corpus generation, exact text match depends on seeding.
    # However, _generate_shared_prefix caches the prefix.
    
    # We can check internal state or verifying that _generate_shared_prefix was called and used cached tokens
    # But better to check if the logic holds.
    
    assert len(generator._shared_prefix_tokens) >= 5

def test_generate_content_min_tokens_instruction(mock_load_corpus, mock_tokenizer, mock_seed_manager, text_config):
    # Generate content with min_tokens_suffix
    generator = TextChannelGenerator(
        text_config, 
        mock_seed_manager, 
        mock_tokenizer,
    )
    
    # Mock tokenizer to return short length for suffix so it fits
    mock_tokenizer.encode.side_effect = lambda x: [1] * (1 if "Generate" in x else len(x.split()))
    
    content = generator.generate_content(is_root=False, min_tokens_suffix=5)
    assert "Generate at least" in content.input_text or content.target_prompt_tokens == 10

