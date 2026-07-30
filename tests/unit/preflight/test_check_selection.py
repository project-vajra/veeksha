"""Which checks a config selects.

``--checks`` is positive selection: naming checks runs exactly those, and naming
none runs them all.
"""

import pytest

from veeksha.config.preflight import PREFLIGHT_CHECKS, PreflightCheckConfig


@pytest.mark.unit
def test_no_selection_runs_every_check():
    assert PreflightCheckConfig().selected_checks() == PREFLIGHT_CHECKS


@pytest.mark.unit
def test_named_checks_are_the_only_ones_run():
    config = PreflightCheckConfig(checks=["chat", "stt"])

    assert config.selected_checks() == ("chat", "stt")


@pytest.mark.unit
def test_selection_order_and_repeats_do_not_change_the_run():
    reordered = PreflightCheckConfig(checks=["stt", "chat", "stt"])

    # Canonical order, deduplicated, so the report layout is stable.
    assert reordered.selected_checks() == ("chat", "stt")


@pytest.mark.unit
def test_unknown_check_is_rejected_with_the_valid_names():
    with pytest.raises(ValueError, match="Unknown preflight check"):
        PreflightCheckConfig(checks=["chat", "nope"])
