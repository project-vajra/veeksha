import json
import os

import pytest

from veeksha.sweep_summary import write_sweep_summary


@pytest.mark.unit
def test_write_sweep_summary_aggregates_decode_window_metrics(tmp_path) -> None:
    base = tmp_path / "decode_sweep"
    base.mkdir()

    run1 = base / "run1"
    run2 = base / "run2"
    (run1 / "metrics").mkdir(parents=True)
    (run2 / "metrics").mkdir(parents=True)

    # Minimal config.yml for context length inference.
    (run1 / "config.yml").write_text(
        """
traffic_scheduler:
  type: concurrent
  target_concurrent_sessions: 4
session_generator:
  channels:
    - type: text
      body_length_generator:
        type: fixed
        value: 512
""".lstrip()
    )
    (run2 / "config.yml").write_text(
        """
traffic_scheduler:
  type: concurrent
  target_concurrent_sessions: 8
session_generator:
  channels:
    - type: text
      body_length_generator:
        type: fixed
        value: 1024
""".lstrip()
    )

    # Prefill stats with single group (helps inference).
    (run1 / "metrics" / "prefill_stats.json").write_text(
        json.dumps({"groups": {"512": {"count": 1}}})
    )
    (run2 / "metrics" / "prefill_stats.json").write_text(
        json.dumps({"groups": {"1024": {"count": 1}}})
    )

    # Decode window metrics (what we aggregate).
    (run1 / "metrics" / "decode_window_metrics.json").write_text(
        json.dumps(
            {
                "config": {"min_active_requests": 4},
                "window": {"start": 0.1, "end": 0.2},
                "tbc_in_window_stats": {"count": 10, "mean": 0.01},
            }
        )
    )
    (run2 / "metrics" / "decode_window_metrics.json").write_text(
        json.dumps(
            {
                "config": {"min_active_requests": 8},
                "window": {"start": 0.1, "end": 0.2},
                "tbc_in_window_stats": {"count": 20, "mean": 0.02},
            }
        )
    )

    # Optional standard artifacts (ensure sweep_summary includes them when present).
    (run1 / "metrics" / "summary_stats.json").write_text(
        json.dumps({"Number of Requests": 10, "Number of Completed Requests": 10, "Error Rate": 0.0})
    )
    (run2 / "metrics" / "summary_stats.json").write_text(
        json.dumps({"Number of Requests": 20, "Number of Completed Requests": 20, "Error Rate": 0.0})
    )
    (run1 / "metrics" / "throughput_metrics.json").write_text(
        json.dumps({"tpot_based_throughput": 100.0, "tbc_based_throughput": 90.0})
    )
    (run2 / "metrics" / "throughput_metrics.json").write_text(
        json.dumps({"tpot_based_throughput": 200.0, "tbc_based_throughput": 180.0})
    )
    (run1 / "metrics" / "slo_results.json").write_text(
        json.dumps({"all_slos_met": True, "results": []})
    )
    (run2 / "metrics" / "slo_results.json").write_text(
        json.dumps({"all_slos_met": True, "results": []})
    )

    written = write_sweep_summary(
        str(base),
        [str(run1), str(run2)],
    )

    assert os.path.exists(written["sweep_manifest"])
    assert os.path.exists(written["sweep_summary"])
    assert os.path.exists(written["sweep_summary_csv"])
    assert os.path.exists(written["decode_stats"])
    assert os.path.exists(written["decode_stats_csv"])

    decode = json.loads((base / "decode_stats.json").read_text())
    assert "512_4" in decode
    assert "1024_8" in decode
    assert decode["512_4"]["tbc_in_window_stats"]["mean"] == pytest.approx(0.01)
    assert decode["1024_8"]["tbc_in_window_stats"]["mean"] == pytest.approx(0.02)


