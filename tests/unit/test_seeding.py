
import pytest

from veeksha.config.generator.interval import PoissonIntervalGeneratorConfig
from veeksha.config.generator.length import UniformLengthGeneratorConfig, FixedLengthGeneratorConfig
from veeksha.config.generator.requested_output import OutputSpecConfig, TextOutputSpecConfig
from veeksha.config.generator.session import SyntheticSessionGeneratorConfig
from veeksha.config.generator.session_graph import LinearSessionGraphGeneratorConfig
from veeksha.config.generator.channel import TextChannelGeneratorConfig
from veeksha.generator.interval.registry import IntervalGeneratorRegistry
from veeksha.generator.length.registry import LengthGeneratorRegistry
from veeksha.generator.session.synthetic import SyntheticSessionGenerator
from veeksha.core.seeding import SeedManager, derive_seed
from veeksha.core.tokenizer import TokenizerHandle, TokenizerProvider
from veeksha.types import ChannelModality

# Mock Tokenizer (copied/adapted)
def mock_tokenizer_handle() -> TokenizerHandle[str]:
    return TokenizerHandle(
        count_tokens=lambda x: len(str(x).split()), # simplistic
        decode=lambda x: "".join([chr(t) for t in x]),
        encode=lambda x: [ord(c) for c in x],
        get_vocab=lambda: list(range(128)) 
    )

@pytest.fixture
def tokenizer_provider() -> TokenizerProvider:
    return TokenizerProvider(
        tokenizers={ChannelModality.TEXT: mock_tokenizer_handle()}
    )

@pytest.mark.unit
class TestSeeding:
    """Test that seed propagation works correctly."""

    def test_seed_derivation_deterministic(self):
        assert derive_seed(123, "foo", "bar") == derive_seed(123, "foo", "bar")

    def test_seed_derivation_differs_with_path(self):
        assert derive_seed(123, "foo", "bar") != derive_seed(123, "foo", "baz")

    def test_seed_derivation_differs_with_root(self):
        assert derive_seed(123, "foo") != derive_seed(456, "foo")

    def test_seed_manager_produces_stable_factories(self):
        manager = SeedManager(999)
        factory = manager.numpy_factory("interval")

        seq_first = [factory().random() for _ in range(3)]
        seq_second = [factory().random() for _ in range(3)]

        manager_again = SeedManager(999)
        factory_again = manager_again.numpy_factory("interval")
        seq_first_again = [factory_again().random() for _ in range(3)]

        assert seq_first == seq_first_again
        assert seq_first != seq_second

    def test_interval_generator_uses_seed_manager(self):
        config = PoissonIntervalGeneratorConfig(arrival_rate=10.0) # arrival_rate replaces qps
        manager = SeedManager(555)

        generator = IntervalGeneratorRegistry.get(
            config.get_type(), config=config, rng=manager.numpy_factory("interval")()
        )
        values = [generator.get_next_interval() for _ in range(3)]

        generator2 = IntervalGeneratorRegistry.get(
            config.get_type(), config=config, rng=SeedManager(555).numpy_factory("interval")()
        )
        values2 = [generator2.get_next_interval() for _ in range(3)]

        assert values == values2

    def test_length_generator_uses_seed_manager(self):
        config = UniformLengthGeneratorConfig(
            min=10, max=20 # properties renamed from min/max_tokens
        )
        manager = SeedManager(777)

        generator = LengthGeneratorRegistry.get(
            config.get_type(), config=config, rng=manager.numpy_factory("length")()
        )
        values = [generator.get_next_value() for _ in range(3)]

        generator2 = LengthGeneratorRegistry.get(
            config.get_type(), config=config, rng=SeedManager(777).numpy_factory("length")()
        )
        values2 = [generator2.get_next_value() for _ in range(3)]

        assert values == values2

    def test_synthetic_generator_reproducibility(self, tokenizer_provider):
        manager = SeedManager(1234)

        # Config that produces 1 request per session
        config = SyntheticSessionGeneratorConfig(
            session_graph=LinearSessionGraphGeneratorConfig(
                num_request_generator=FixedLengthGeneratorConfig(value=1)
            ),
            channels=[
                TextChannelGeneratorConfig(
                    body_length_generator=UniformLengthGeneratorConfig(min=5, max=5),
                )
            ],
            output_spec=OutputSpecConfig(
                text=TextOutputSpecConfig(
                    output_length_generator=FixedLengthGeneratorConfig(value=10)
                )
            )
        )

        generator = SyntheticSessionGenerator(
            config=config,
            tokenizer_provider=tokenizer_provider,
            seed_manager=manager,
        )

        sessions = []
        for _ in range(3):
            sessions.append(generator.generate_session())

        generator2 = SyntheticSessionGenerator(
            config=config,
            tokenizer_provider=tokenizer_provider, 
            seed_manager=SeedManager(1234),
        )

        sessions2 = []
        for _ in range(3):
            sessions2.append(generator2.generate_session())

        for s1, s2 in zip(sessions, sessions2):
            assert s1.id == s2.id 
            req1 = s1.requests[0]
            req2 = s2.requests[0]
            
            # Check content equality (depends on seeding)
            assert req1.channels[ChannelModality.TEXT].input_text == req2.channels[ChannelModality.TEXT].input_text
            
            # Check timing (wait_after_ready)
            # Find the node ID for the request (it's 0 usually but let's be safe)
            node_id = list(s1.requests.keys())[0] # assume 1 request
            assert s1.session_graph.nodes[node_id].wait_after_ready == s2.session_graph.nodes[node_id].wait_after_ready
