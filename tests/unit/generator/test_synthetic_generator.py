"""Unit tests for SyntheticSessionGenerator."""

from typing import List

import pytest

from veeksha.config.generator.channel import TextChannelGeneratorConfig
from veeksha.config.generator.length import FixedLengthGeneratorConfig
from veeksha.config.generator.requested_output import OutputSpecConfig, TextOutputSpecConfig
from veeksha.config.generator.session import SyntheticSessionGeneratorConfig
from veeksha.config.generator.session_graph import LinearSessionGraphGeneratorConfig
from veeksha.core.seeding import SeedManager
from veeksha.core.tokenizer import TokenizerHandle, TokenizerProvider
from veeksha.generator.session.synthetic import SyntheticSessionGenerator
from veeksha.types import ChannelModality


def mock_tokenizer_handle() -> TokenizerHandle[str]:
    return TokenizerHandle(
        count_tokens=lambda x: len(x.split()),
        decode=lambda x: " ".join([str(t) for t in x]),
        encode=lambda x: [1] * len(x.split()), # Mock encode
        get_vocab=lambda: [1, 2, 3] # Mock vocab
    )

@pytest.fixture
def tokenizer_provider() -> TokenizerProvider:
    return TokenizerProvider(
        tokenizers={ChannelModality.TEXT: mock_tokenizer_handle()}
    )

@pytest.fixture
def linear_session_config() -> SyntheticSessionGeneratorConfig:
    return SyntheticSessionGeneratorConfig(
        session_graph=LinearSessionGraphGeneratorConfig(
            num_request_generator=FixedLengthGeneratorConfig(value=2)
        ),
        channels=[
            TextChannelGeneratorConfig(
                body_length_generator=FixedLengthGeneratorConfig(value=10),
            )
        ],
        output_spec=OutputSpecConfig(
            text=TextOutputSpecConfig(
                output_length_generator=FixedLengthGeneratorConfig(value=5)
            )
        )
    )

@pytest.mark.unit
def test_generate_simple_session(tokenizer_provider, linear_session_config) -> None:
    seed_manager = SeedManager(seed=42)
    generator = SyntheticSessionGenerator(
        config=linear_session_config, 
        seed_manager=seed_manager,
        tokenizer_provider=tokenizer_provider
    )
    
    session = generator.generate_session()
    
    # Check session structure
    assert len(session.requests) == 2
    assert session.session_graph.nodes.keys() == {0, 1}
    
    # Check request content
    req0 = session.requests[0]
    assert ChannelModality.TEXT in req0.channels
    
    # Check that output spec is attached
    assert req0.requested_output is not None
    assert req0.requested_output.text is not None
    assert req0.requested_output.text.target_tokens == 5

@pytest.mark.unit
def test_generate_session_ids_increment(tokenizer_provider, linear_session_config) -> None:
    seed_manager = SeedManager(seed=42)
    generator = SyntheticSessionGenerator(
        config=linear_session_config, 
        seed_manager=seed_manager,
        tokenizer_provider=tokenizer_provider
    )
    
    s1 = generator.generate_session()
    s2 = generator.generate_session()
    
    assert s1.id == 0
    assert s2.id == 1
    
    # Check request IDs
    # s1 has 2 requests (req_ids 0, 1)
    # s2 has 2 requests (req_ids 2, 3)
    assert list(s1.requests.values())[0].id == 0
    assert list(s1.requests.values())[1].id == 1
    assert list(s2.requests.values())[0].id == 2
