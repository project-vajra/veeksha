import pytest

from veeksha.config.client import ClientConfig
from veeksha.config.generators.interval_generator.poisson_generator import (
    PoissonRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.length_generator.uniform_generator import (
    UniformRequestLengthGeneratorConfig,
)
from veeksha.config.generators.request_generator.synthetic_generator import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.generators.interval_generator.generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from veeksha.generators.length_generator.generator_registry import (
    RequestLengthGeneratorRegistry,
)
from veeksha.generators.request_generator.synthetic_generator import (
    SyntheticRequestGenerator,
)
from veeksha.core.seeding import SeedManager, derive_seed


class DummyTokenizer:
    """Simple deterministic tokenizer for tests."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(ch) for ch in text]

    def decode(self, tokens, skip_special_tokens: bool = False):
        return "".join(chr(int(t)) for t in tokens)


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
        config = PoissonRequestIntervalGeneratorConfig(qps=10.0)
        manager = SeedManager(555)

        generator = RequestIntervalGeneratorRegistry.get(
            config.get_type(), config=config, rng=manager.numpy_factory("interval")()
        )
        values = [generator.get_next_inter_request_time() for _ in range(3)]

        generator2 = RequestIntervalGeneratorRegistry.get(
            config.get_type(), config=config, rng=SeedManager(555).numpy_factory("interval")()
        )
        values2 = [generator2.get_next_inter_request_time() for _ in range(3)]

        assert values == values2


    def test_length_generator_uses_seed_manager(self):
        config = UniformRequestLengthGeneratorConfig(
            min_tokens=10, max_tokens=20, prefill_to_decode_ratio=1.0
        )
        manager = SeedManager(777)

        generator = RequestLengthGeneratorRegistry.get(
            config.get_type(), config=config, rng=manager.numpy_factory("length")()
        )
        values = [generator.get_next_num_tokens() for _ in range(3)]

        generator2 = RequestLengthGeneratorRegistry.get(
            config.get_type(), config=config, rng=SeedManager(777).numpy_factory("length")()
        )
        values2 = [generator2.get_next_num_tokens() for _ in range(3)]

        assert values == values2


    def test_synthetic_generator_reproducibility(self):
        tokenizer = DummyTokenizer()
        manager = SeedManager(1234)

        config = SyntheticRequestGeneratorConfig(
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=2.0),
            length_generator_config=UniformRequestLengthGeneratorConfig(
                min_tokens=5, max_tokens=5, prefill_to_decode_ratio=1.0
            ),
        )

        generator = SyntheticRequestGenerator(
            config=config,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            client_config=ClientConfig(),
            seed_manager=manager,
            corpus_lines=["hello world"],
        )

        requests = []
        for _ in range(3):
            try:
                requests.append(generator.get_request())
            except StopIteration:
                pytest.fail("Generator exhausted before producing 3 requests")

        generator2 = SyntheticRequestGenerator(
            config=config,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            client_config=ClientConfig(),
            seed_manager=SeedManager(1234),
            corpus_lines=["hello world"],
        )

        requests2 = []
        for _ in range(3):
            try:
                requests2.append(generator2.get_request())
            except StopIteration:
                pytest.fail("Generator exhausted before producing 3 requests (second generator)")

        for req1, req2 in zip(requests, requests2):
            assert req1.prompt == req2.prompt
            assert req1.session_start_time == req2.session_start_time
            assert (
                req1.wait_after_prev_response_s == req2.wait_after_prev_response_s
            )
