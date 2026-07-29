from __future__ import annotations

import pytest

from veeksha.evaluator.cdf_sketch import CDFSketch


@pytest.mark.unit
def test_empty_sketch_reports_no_keys() -> None:
    """A metric that was never observed must not report 0.

    Fabricated zeros read as a perfect result: a run where every request
    failed would otherwise summarize as 0 ms mean and 0 ms P99.
    """
    sketch = CDFSketch("ttfc", unit="ms")

    assert sketch.get_summary() == {}
    assert "no samples" in str(sketch)


@pytest.mark.unit
def test_populated_sketch_reports_mean_and_percentiles() -> None:
    sketch = CDFSketch("ttfc", unit="ms")
    sketch.extend([100.0, 200.0, 300.0])

    summary = sketch.get_summary()

    assert set(summary) == {
        "ttfc (Mean)",
        "ttfc (P50)",
        "ttfc (P90)",
        "ttfc (P99)",
    }
    assert summary["ttfc (Mean)"] == pytest.approx(200.0)
    assert summary["ttfc (P50)"] == pytest.approx(200.0, rel=1e-2)
    assert "no samples" not in str(sketch)


@pytest.mark.unit
def test_zero_valued_sample_still_reports_keys() -> None:
    """A real 0 ms observation is distinct from no observation at all."""
    sketch = CDFSketch("ttfc", unit="ms")
    sketch.put(0.0)

    summary = sketch.get_summary()

    assert summary != {}
    assert summary["ttfc (Mean)"] == pytest.approx(0.0)
