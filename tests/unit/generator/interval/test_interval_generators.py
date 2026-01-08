
import pytest
import numpy as np
from veeksha.config.generator.interval import (
    FixedIntervalGeneratorConfig,
    GammaIntervalGeneratorConfig,
    PoissonIntervalGeneratorConfig,
)
from veeksha.generator.interval.fixed import FixedIntervalGenerator
from veeksha.generator.interval.gamma import GammaIntervalGenerator
from veeksha.generator.interval.poisson import PoissonIntervalGenerator

@pytest.fixture
def rng():
    return np.random.RandomState(42)

def test_fixed_interval_generator():
    config = FixedIntervalGeneratorConfig(interval=1.5)
    generator = FixedIntervalGenerator(config, rng=None)
    assert generator.get_next_interval() == 1.5

def test_gamma_interval_generator(rng):
    config = GammaIntervalGeneratorConfig(arrival_rate=10.0, cv=1.0)
    generator = GammaIntervalGenerator(config, rng)
    
    # Check a few values
    values = [generator.get_next_interval() for _ in range(100)]
    assert all(v > 0 for v in values)
    # Mean should be approx 1/arrival_rate = 0.1
    assert abs(np.mean(values) - 0.1) < 0.05

def test_poisson_interval_generator(rng):
    config = PoissonIntervalGeneratorConfig(arrival_rate=5.0)
    generator = PoissonIntervalGenerator(config, rng)
    
    values = [generator.get_next_interval() for _ in range(100)]
    assert all(v > 0 for v in values)
    # Mean should be approx 1/arrival_rate = 0.2
    assert abs(np.mean(values) - 0.2) < 0.05
    
    # Check max interval constraint
    assert all(v <= generator.max_interval for v in values)
