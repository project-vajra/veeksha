import pytest

from veeksha.config.generator.length import StairLengthGeneratorConfig
from veeksha.generator.length.stair import StairLengthGenerator


@pytest.mark.unit
def test_stair_length_generator_repeats_and_wraps() -> None:
    cfg = StairLengthGeneratorConfig(values=[1, 2, 3], repeat_each=2, wrap=True)
    gen = StairLengthGenerator(cfg)

    seq = [gen.get_next_value() for _ in range(8)]
    assert seq == [1, 1, 2, 2, 3, 3, 1, 1]


@pytest.mark.unit
def test_stair_length_generator_can_hold_last_value_without_wrap() -> None:
    cfg = StairLengthGeneratorConfig(values=[5, 7], repeat_each=1, wrap=False)
    gen = StairLengthGenerator(cfg)

    seq = [gen.get_next_value() for _ in range(5)]
    assert seq == [5, 7, 7, 7, 7]





