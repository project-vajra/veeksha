"""CLI entry point for ``veeksha preflight``.

Runs each configured check, scores the timing drift, gates it into a verdict,
prints and writes a per-check report (under ``<output_dir>/<check>/``), and
exits non-zero unless every check comes back honest.
"""

from __future__ import annotations

import os
import sys
from typing import List

from veeksha.config.preflight import PreflightCheckConfig
from veeksha.logger import init_logger
from veeksha.preflight import validator
from veeksha.preflight.drivers import (
    run_completions_preflight,
    run_streaming_tts_openai_preflight,
    run_streaming_tts_vajra_preflight,
    run_stt_preflight,
    run_text_preflight,
    run_tts_preflight,
)
from veeksha.preflight.models import ScoreReport
from veeksha.preflight.report import render_report, write_report

logger = init_logger(__name__)


def _gate_and_render(
    config: PreflightCheckConfig,
    report: ScoreReport,
    title: str,
    output_dir: str,
    check_config,
) -> bool:
    """Validate one check's report, print + write it, return True iff PASS."""
    result = validator.run_validation(
        report,
        delivery_lag_threshold_ms=config.delivery_lag_threshold_ms,
        server_pacing_threshold_ms=config.server_pacing_threshold_ms,
        dispatch_drift_threshold_ms=config.dispatch_drift_threshold_ms,
        input_pacing_threshold_ms=config.input_pacing_threshold_ms,
        max_unpaired_fraction=config.max_unpaired_fraction,
    )
    text = render_report(report, result, config, check_config, title=title)
    path = write_report(text, output_dir)
    print(text)
    print(f"Report written to {path}")
    return result.is_pass


def _run_config(config: PreflightCheckConfig, output_dir: str) -> bool:
    """Run every enabled check for one config; return True iff all passed.

    Each check gets its own ``<output_dir>/<slug>/`` subdirectory (report +
    metrics) so the per-modality outputs don't overwrite one another.
    """
    all_passed = True
    common = dict(
        traffic_scheduler=config.build_traffic(),
        runtime=config.runtime,
    )

    # check name -> (runner, group config, display name)
    checks = {
        "chat": (run_text_preflight, config.text, "text (chat)"),
        "completions": (run_completions_preflight, config.text, "completions"),
        "tts": (run_tts_preflight, config.tts, "tts"),
        "streaming_tts_openai": (
            run_streaming_tts_openai_preflight,
            config.tts,
            "streaming_tts (openai_realtime)",
        ),
        "streaming_tts_vajra": (
            run_streaming_tts_vajra_preflight,
            config.tts,
            "streaming_tts (vajra)",
        ),
        "stt": (run_stt_preflight, config.stt, "stt"),
    }
    for slug in config.selected_checks():
        runner, group_cfg, name = checks[slug]
        logger.info("Running preflight %s check", name)
        check_output_dir = os.path.join(output_dir, slug)
        report = runner(group_cfg, output_dir=check_output_dir, **common)
        all_passed &= _gate_and_render(
            config,
            report,
            f"Preflight measurement fidelity: {name}",
            check_output_dir,
            check_config=group_cfg,
        )

    return all_passed


def run_preflight_cli(configs: List[PreflightCheckConfig]) -> None:
    overall_passed = True
    # A sweep's configs share one output_dir; number them so their reports don't
    # overwrite each other.
    is_sweep = len(configs) > 1
    for index, config in enumerate(configs):
        output_dir = config.output_dir
        if is_sweep:
            output_dir = os.path.join(output_dir, f"run_{index:03d}")
        overall_passed = _run_config(config, output_dir) and overall_passed

    if not overall_passed:
        # Non-zero exit so CI/scripts can gate a benchmark on a clean preflight.
        # FAIL and SERVER_AT_CAPACITY both count as "did not pass".
        sys.exit(1)
