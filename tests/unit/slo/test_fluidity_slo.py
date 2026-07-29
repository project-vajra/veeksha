from veeksha.config.slo import ConstantSloConfig
from veeksha.slo.slo import ConstantSlo


def test_fluidity_slo_uses_lower_tail_and_higher_is_better() -> None:
    slo = ConstantSlo(
        ConstantSloConfig(
            metric="user_audio_fluidity_index",
            percentile=0.01,
            value=0.90,
        )
    )

    met, observed = slo.evaluate({"user_audio_fluidity_index": [0.95, 0.97, 0.99, 1.0]})

    assert met
    assert observed > 0.90


def test_fluidity_slo_fails_when_lower_tail_is_below_threshold() -> None:
    slo = ConstantSlo(
        ConstantSloConfig(
            metric="user_audio_fluidity_index",
            percentile=0.01,
            value=0.99,
        )
    )

    met, observed = slo.evaluate({"user_audio_fluidity_index": [0.80, 0.99, 1.0, 1.0]})

    assert not met
    assert observed < 0.99
