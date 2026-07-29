"""Unit tests for declared free variables (knobs)."""

from __future__ import annotations

import argparse

import pytest

from veeksha.benchmark_knobs import (
    KnobDeclarationError,
    add_knob_arguments,
    apply_knobs,
    build_knobs_config_class,
    parse_knob_specs,
    resolve_knob_values,
)


def _sample_raw():
    return {
        "concurrency": {
            "target": "traffic_scheduler.target_concurrent_sessions",
            "type": "int",
            "default": 1,
            "choices": [1, 8, 16, 32],
            "help": "Steady-state concurrent sessions.",
        },
        "input_tps": {
            "target": "client.pacing.tokens_per_second",
            "type": "float",
            "default": 50.0,
            "help": "Text pacing rate.",
        },
    }


@pytest.mark.unit
def test_parse_knob_specs_and_defaults() -> None:
    specs = parse_knob_specs(_sample_raw())
    assert [s.name for s in specs] == ["concurrency", "input_tps"]
    values = resolve_knob_values(specs)
    assert values == {"concurrency": 1, "input_tps": 50.0}


@pytest.mark.unit
def test_choices_reject_out_of_range() -> None:
    specs = parse_knob_specs(_sample_raw())
    with pytest.raises(KnobDeclarationError, match="not one of"):
        resolve_knob_values(specs, {"concurrency": 99})


@pytest.mark.unit
def test_build_knobs_config_class() -> None:
    specs = parse_knob_specs(_sample_raw())
    cls = build_knobs_config_class(specs)
    instance = cls()
    assert instance.concurrency == 1
    assert instance.input_tps == 50.0
    fields = instance.__dataclass_fields__
    assert "concurrency" in fields
    help_text = fields["concurrency"].metadata.get("help", "")
    assert "concurrent" in help_text.lower() or "Steady-state" in help_text


@pytest.mark.unit
def test_add_knob_arguments_help_and_choices() -> None:
    specs = parse_knob_specs(_sample_raw())
    parser = argparse.ArgumentParser()
    add_knob_arguments(parser, specs)
    help_text = parser.format_help()
    assert "--concurrency" in help_text
    assert "--input_tps" in help_text
    assert "Steady-state" in help_text


@pytest.mark.unit
def test_apply_knobs_targets_dotted_paths() -> None:
    specs = parse_knob_specs(_sample_raw())
    values = resolve_knob_values(specs, {"concurrency": 16, "input_tps": 25.0})
    config = {
        "traffic_scheduler": {"target_concurrent_sessions": 1, "other": True},
        "client": {"pacing": {"tokens_per_second": 50.0}},
    }
    merged = apply_knobs(config, specs, values)
    assert merged["traffic_scheduler"]["target_concurrent_sessions"] == 16
    assert merged["traffic_scheduler"]["other"] is True
    assert merged["client"]["pacing"]["tokens_per_second"] == 25.0


@pytest.mark.unit
def test_missing_default_rejected() -> None:
    raw = {
        "concurrency": {
            "target": "traffic_scheduler.target_concurrent_sessions",
            "type": "int",
        }
    }
    with pytest.raises(KnobDeclarationError, match="missing keys"):
        parse_knob_specs(raw)


@pytest.mark.unit
def test_affects_workload_no_longer_accepted() -> None:
    raw = {
        "concurrency": {
            "target": "traffic_scheduler.target_concurrent_sessions",
            "type": "int",
            "default": 1,
            "affects_workload": False,
        }
    }
    with pytest.raises(KnobDeclarationError, match="unknown keys"):
        parse_knob_specs(raw)


@pytest.mark.unit
def test_duplicate_targets_rejected() -> None:
    raw = {
        "a": {
            "target": "x.y",
            "type": "int",
            "default": 1,
        },
        "b": {
            "target": "x.y",
            "type": "int",
            "default": 2,
        },
    }
    with pytest.raises(KnobDeclarationError, match="both target"):
        parse_knob_specs(raw)


@pytest.mark.unit
def test_unknown_knob_name_rejected() -> None:
    specs = parse_knob_specs(_sample_raw())
    with pytest.raises(KnobDeclarationError, match="unknown free variable"):
        resolve_knob_values(specs, {"not_a_knob": 1})
