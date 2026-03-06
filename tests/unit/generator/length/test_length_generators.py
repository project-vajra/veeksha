
import pytest
import numpy as np
from veeksha.config.generator.length import (
    FixedLengthGeneratorConfig,
    InverseGaussianLengthGeneratorConfig,
    UniformLengthGeneratorConfig,
    ZipfLengthGeneratorConfig,
)
from veeksha.generator.length.fixed import FixedLengthGenerator
from veeksha.generator.length.inverse_gaussian import InverseGaussianLengthGenerator
from veeksha.generator.length.uniform import UniformLengthGenerator
from veeksha.generator.length.zipf import ZipfLengthGenerator

@pytest.fixture
def rng():
    return np.random.RandomState(42)

def test_fixed_length_generator():
    config = FixedLengthGeneratorConfig(value=10)
    generator = FixedLengthGenerator(config)
    assert generator.get_next_value() == 10

def test_uniform_length_generator(rng):
    config = UniformLengthGeneratorConfig(min=5, max=10)
    generator = UniformLengthGenerator(config, rng)
    
    values = [generator.get_next_value() for _ in range(100)]
    assert all(5 <= v < 10 for v in values)
    # verify if max is inclusive or exclusive based on implementation details or desired behavior.
    # Assuming standard numpy behavior, it's exclusive of max.
    
    assert min(values) >= 5
    assert max(values) < 10

def test_zipf_length_generator(rng):
    config = ZipfLengthGeneratorConfig(min=1, max=100, theta=1.5, scramble=False)
    generator = ZipfLengthGenerator(config, rng)
    
    values = [generator.get_next_value() for _ in range(100)]
    assert all(1 <= v <= 100 for v in values)
    # Zipf distribution creates many small values
    assert np.mean(values) < 50

def test_zipf_length_generator_scramble(rng):
    config = ZipfLengthGeneratorConfig(min=1, max=100, theta=1.5, scramble=True)
    generator = ZipfLengthGenerator(config, rng)
    
    values = [generator.get_next_value() for _ in range(100)]
    assert all(1 <= v <= 100 for v in values)


def test_inverse_gaussian_length_generator_reproducible():
    config = InverseGaussianLengthGeneratorConfig(mean=12.0, shape=5.0)
    generator = InverseGaussianLengthGenerator(config, np.random.RandomState(42))
    generator2 = InverseGaussianLengthGenerator(config, np.random.RandomState(42))

    values = [generator.get_next_value() for _ in range(20)]
    values2 = [generator2.get_next_value() for _ in range(20)]

    assert values == values2
    assert all(v >= 1 for v in values)


def test_inverse_gaussian_length_generator_config_validation():
    with pytest.raises(ValueError, match="mean must be > 0"):
        InverseGaussianLengthGeneratorConfig(mean=0.0, shape=1.0)

    with pytest.raises(ValueError, match="shape must be > 0"):
        InverseGaussianLengthGeneratorConfig(mean=1.0, shape=0.0)
