
import pytest
import numpy as np
from veeksha.config.generator.length import (
    FixedLengthGeneratorConfig,
    UniformLengthGeneratorConfig,
    ZipfLengthGeneratorConfig,
)
from veeksha.generator.length.fixed import FixedLengthGenerator
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
    assert all(5 <= v < 10 for v in values) # numpy uniform is [low, high) for floats, but let's check exact implementation
    # Implementation: int(rng.uniform(min, max)). So [min, max).
    # Wait, python's random.uniform includes max, but numpy.random.uniform excludes max.
    # The implementation uses self.rng.uniform(self.config.min, self.config.max).
    # We should verify if max is inclusive or exclusive based on implementation details or desired behavior.
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
