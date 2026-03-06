"""Post-run validation for microbenchmark results.

Reads request_level_metrics.jsonl from benchmark output directories and
checks invariants specific to each benchmark type (FCFS ordering, expected
session counts, token counts, decode window coverage, etc.).
"""

import json
import logging
import math
from pathlib import Path

from veeksha.microbench.config import MicrobenchmarkConfig
from veeksha.microbench.config_builder import compute_prefill_iterations

logger = logging.getLogger(__name__)


class ValidationResult:
    """Accumulates pass/warn/fail checks."""

    def __init__(self) -> None:
        self.checks: list[tuple[str, str, str]] = []  # (status, name, detail)

    def passed(self, name: str, detail: str = "") -> None:
        self.checks.append(("PASS", name, detail))

    def warn(self, name: str, detail: str) -> None:
        self.checks.append(("WARN", name, detail))

    def fail(self, name: str, detail: str) -> None:
        self.checks.append(("FAIL", name, detail))

    @property
    def ok(self) -> bool:
        return all(status != "FAIL" for status, _, _ in self.checks)

    def summary(self) -> str:
        lines = []
        for status, name, detail in self.checks:
            label = {"PASS": "  PASS", "WARN": "  WARN", "FAIL": "**FAIL"}[status]
            line = f"{label}  {name}"
            if detail:
                line += f" — {detail}"
            lines.append(line)
        num_passed = sum(1 for status, _, _ in self.checks if status == "PASS")
        num_warnings = sum(1 for status, _, _ in self.checks if status == "WARN")
        num_failures = sum(1 for status, _, _ in self.checks if status == "FAIL")
        lines.append(
            f"\n{num_passed} passed, {num_warnings} warnings, {num_failures} failures"
        )
        return "\n".join(lines)


def _load_request_metrics(output_dir: str) -> list[dict] | None:
    """Find and load request_level_metrics.jsonl from a benchmark output dir."""
    base = Path(output_dir)
    candidates = sorted(base.glob("**/request_level_metrics.jsonl"))
    if not candidates:
        return None
    # Use the most recent run
    path = candidates[-1]
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_decode_window_json(output_dir: str) -> dict | None:
    """Find and load decode_window_metrics.json from a benchmark output dir."""
    base = Path(output_dir)
    candidates = sorted(base.glob("**/decode_window_metrics.json"))
    if not candidates:
        return None
    path = candidates[-1]
    with open(path) as f:
        return json.load(f)


def validate(cfg: MicrobenchmarkConfig, output_dir: str) -> ValidationResult:
    """Run all validations for a microbenchmark output."""
    if cfg.type == "prefill":
        return _validate_prefill(cfg, output_dir)
    elif cfg.type == "decode":
        return _validate_decode(cfg, output_dir)
    elif cfg.type == "mixed":
        return _validate_mixed(cfg, output_dir)
    else:
        result = ValidationResult()
        result.fail("unknown_type", f"No validator for type '{cfg.type}'")
        return result


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


def _validate_prefill(cfg: MicrobenchmarkConfig, output_dir: str) -> ValidationResult:
    result = ValidationResult()
    metrics = _load_request_metrics(output_dir)
    if metrics is None:
        result.fail("metrics_found", "No request_level_metrics.jsonl found")
        return result
    result.passed("metrics_found")

    expected_count = len(cfg.input_lengths) * cfg.samples_per_length

    # Session count
    if len(metrics) == expected_count:
        result.passed("session_count", f"{len(metrics)} requests")
    else:
        result.fail("session_count", f"expected {expected_count}, got {len(metrics)}")

    # No errors
    failed_requests = [
        record for record in metrics if record.get("num_output_tokens", 0) == 0
    ]
    if not failed_requests:
        result.passed("no_errors", "all requests produced output")
    else:
        result.fail(
            "no_errors", f"{len(failed_requests)} requests produced 0 output tokens"
        )

    # Output tokens = expected
    mismatched_output_requests = [
        record for record in metrics if record["num_output_tokens"] != cfg.output_tokens
    ]
    if not mismatched_output_requests:
        result.passed(
            "output_tokens", f"all requests produced {cfg.output_tokens} output tokens"
        )
    else:
        result.warn(
            "output_tokens",
            f"{len(mismatched_output_requests)} requests had unexpected output token count",
        )

    # Sequential execution (concurrent=1): each request dispatched after prior completed
    sorted_by_session = sorted(metrics, key=lambda record: record["session_id"])
    is_sequential = True
    for i in range(1, len(sorted_by_session)):
        previous_completion_time = sorted_by_session[i - 1]["client_completed_at"]
        current_dispatch_time = sorted_by_session[i]["scheduler_dispatched_at"]
        if current_dispatch_time < previous_completion_time - 0.01:  # 10ms tolerance
            is_sequential = False
            break
    if is_sequential:
        result.passed("sequential_execution", "requests executed one at a time")
    else:
        result.warn(
            "sequential_execution", "some requests overlapped (concurrent != 1?)"
        )

    # Prompt tokens match stair pattern
    for i, record in enumerate(sorted_by_session):
        length_idx = i // cfg.samples_per_length
        if length_idx < len(cfg.input_lengths):
            expected_prompt = cfg.input_lengths[length_idx]
            actual_prompt = record["target_num_delta_prompt_tokens"]
            if actual_prompt != expected_prompt:
                result.warn(
                    "prompt_tokens_stair",
                    f"session {record['session_id']}: expected {expected_prompt}, got {actual_prompt}",
                )
                break
    else:
        result.passed(
            "prompt_tokens_stair", "prompt tokens follow expected stair pattern"
        )

    return result


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def _validate_decode(cfg: MicrobenchmarkConfig, output_dir: str) -> ValidationResult:
    result = ValidationResult()

    # Validate each (batch_size, input_length) run
    for batch_size in cfg.batch_sizes:
        for input_length in cfg.input_lengths:
            check_label = f"bs={batch_size},il={input_length}"
            _validate_one_decode_run(
                result, cfg, batch_size, input_length, output_dir, check_label
            )

    return result


def _validate_one_decode_run(
    result: ValidationResult,
    cfg: MicrobenchmarkConfig,
    batch_size: int,
    input_length: int,
    output_dir: str,
    check_label: str,
) -> None:
    # We can't easily map which sub-run dir belongs to which (bs, il) pair
    # from the output_dir alone, so we validate the sweep-level metrics.
    # For now, check the overall sweep directory.
    metrics = _load_request_metrics(f"{output_dir}/bs={batch_size}_il={input_length}")
    if metrics is None:
        result.fail(
            f"metrics_found [{check_label}]",
            "No request_level_metrics.jsonl found in decode dir",
        )
        return

    # Filter to requests matching this input_length
    matching_requests = [
        record
        for record in metrics
        if record["target_num_delta_prompt_tokens"] == input_length
    ]

    if not matching_requests:
        result.warn(
            f"matching_requests [{check_label}]",
            f"no requests with prompt={input_length}",
        )
        return
    result.passed(
        f"matching_requests [{check_label}]", f"{len(matching_requests)} requests"
    )

    # FCFS ordering: first tokens should arrive roughly in session order
    sorted_by_session = sorted(
        matching_requests, key=lambda record: record["session_id"]
    )
    first_token_times = [
        record["client_picked_up_at"] + record["ttfc"] for record in sorted_by_session
    ]
    out_of_order = sum(
        1
        for i in range(1, len(first_token_times))
        if first_token_times[i] < first_token_times[i - 1] - 0.005
    )
    if out_of_order == 0:
        result.passed(
            f"fcfs_order [{check_label}]", "first tokens arrived in session order"
        )
    else:
        result.warn(
            f"fcfs_order [{check_label}]",
            f"{out_of_order} out-of-order first tokens (engine may batch prefills)",
        )

    # Decode window overlap check
    decode_window_data = _load_decode_window_json(
        f"{output_dir}/bs={batch_size}_il={input_length}"
    )
    if decode_window_data is not None:
        num_segments = decode_window_data.get("windows", {}).get(
            "num_selected_segments", 0
        )
        tbc_count = decode_window_data.get("tbc_in_window_stats", {}).get("count", 0)
        if num_segments == 0 or tbc_count == 0:
            result.fail(
                f"decode_window_overlap [{check_label}]",
                "no qualifying decode windows found — increase output tokens",
            )
        elif tbc_count < cfg.samples_per_length:
            result.warn(
                f"decode_window_overlap [{check_label}]",
                f"low sample count in decode window: {tbc_count} < {cfg.samples_per_length}",
            )
        else:
            result.passed(
                f"decode_window_overlap [{check_label}]",
                f"{tbc_count} samples in decode window",
            )


# ---------------------------------------------------------------------------
# Mixed batch
# ---------------------------------------------------------------------------


def _validate_mixed(cfg: MicrobenchmarkConfig, output_dir: str) -> ValidationResult:
    result = ValidationResult()

    for batch_size in cfg.batch_sizes:
        for decode_input_length in cfg.decode_input_lengths:
            for prefill_kv_length in cfg.prefill_kv_lengths:
                for incremental_prefill_size in cfg.incremental_prefill_sizes:
                    check_label = f"bs={batch_size},dil={decode_input_length},kv={prefill_kv_length},dp={incremental_prefill_size}"
                    output_dir_tag = f"bs={batch_size}_dil={decode_input_length}_kv={prefill_kv_length}_dp={incremental_prefill_size}"
                    _validate_one_mixed_run(
                        result,
                        cfg,
                        batch_size,
                        decode_input_length,
                        prefill_kv_length,
                        incremental_prefill_size,
                        output_dir,
                        output_dir_tag,
                        check_label,
                    )

    return result


def _validate_one_mixed_run(
    result: ValidationResult,
    cfg: MicrobenchmarkConfig,
    batch_size: int,
    decode_input_length: int,
    prefill_kv_length: int,
    incremental_prefill_size: int,
    output_dir: str,
    output_dir_tag: str,
    check_label: str,
) -> None:
    # -- Warmup validation --
    warmup_metrics = _load_request_metrics(f"{output_dir}/{output_dir_tag}/warmup")
    if warmup_metrics is None:
        result.warn(f"warmup_found [{check_label}]", "no warmup metrics found")
    else:
        result.passed(
            f"warmup_found [{check_label}]", f"{len(warmup_metrics)} warmup requests"
        )

    # -- Benchmark validation --
    metrics = _load_request_metrics(f"{output_dir}/{output_dir_tag}/bench")
    if metrics is None:
        result.fail(
            f"metrics_found [{check_label}]", "No request_level_metrics.jsonl found"
        )
        return
    result.passed(f"metrics_found [{check_label}]")

    sorted_by_session = sorted(metrics, key=lambda record: record["session_id"])

    # Identify decode vs interference by output token count
    samples_per_prefill = compute_prefill_iterations(
        incremental_prefill_size,
        cfg.engine_chunk_size,
        batch_size,
    )
    num_prefill_requests = math.ceil(cfg.samples_per_length / samples_per_prefill)
    expected_bench_sessions = batch_size + num_prefill_requests

    if len(sorted_by_session) != expected_bench_sessions:
        result.warn(
            f"session_count [{check_label}]",
            f"expected {expected_bench_sessions}, got {len(sorted_by_session)}",
        )
    else:
        result.passed(
            f"session_count [{check_label}]", f"{len(sorted_by_session)} sessions"
        )

    # Split into decode and interference requests based on session order
    decode_requests = sorted_by_session[:batch_size]
    interference_requests = sorted_by_session[batch_size:]

    # Check decode request prompt tokens
    mismatched_decode_prompts = [
        record
        for record in decode_requests
        if record["target_num_delta_prompt_tokens"] != decode_input_length
    ]
    if not mismatched_decode_prompts:
        result.passed(
            f"decode_prompts [{check_label}]",
            f"all decode requests have prompt={decode_input_length}",
        )
    else:
        result.warn(
            f"decode_prompts [{check_label}]",
            f"{len(mismatched_decode_prompts)} decode requests had wrong prompt token count",
        )

    # Check interference request prompt tokens
    expected_interference_prompt_tokens = prefill_kv_length + incremental_prefill_size
    mismatched_interference_prompts = [
        record
        for record in interference_requests
        if record["target_num_delta_prompt_tokens"]
        != expected_interference_prompt_tokens
    ]
    if not mismatched_interference_prompts:
        result.passed(
            f"interference_prompts [{check_label}]",
            f"all interference requests have prompt={expected_interference_prompt_tokens}",
        )
    else:
        result.warn(
            f"interference_prompts [{check_label}]",
            f"{len(mismatched_interference_prompts)} interference requests had wrong prompt token count",
        )

    # FCFS: decode first tokens before interference first tokens
    if decode_requests and interference_requests:
        decode_first_token_times = [
            record["client_picked_up_at"] + record["ttfc"] for record in decode_requests
        ]
        interference_first_token_times = [
            record["client_picked_up_at"] + record["ttfc"]
            for record in interference_requests
        ]
        latest_decode_first_token = max(decode_first_token_times)
        earliest_interference_first_token = min(interference_first_token_times)

        if latest_decode_first_token < earliest_interference_first_token:
            result.passed(
                f"fcfs_decode_before_interference [{check_label}]",
                f"decode last_ft={latest_decode_first_token:.3f} < interference first_ft={earliest_interference_first_token:.3f}",
            )
        else:
            # Warn rather than fail: client-side dispatch ordering is
            # guaranteed by the ticket mechanism, but server-side continuous
            # batching can complete interference prefills before decode
            # prefills, causing first-token reordering.
            result.warn(
                f"fcfs_decode_before_interference [{check_label}]",
                f"decode last_ft={latest_decode_first_token:.3f} >= interference first_ft={earliest_interference_first_token:.3f} "
                f"— server-side batching may reorder first tokens",
            )

    # No errors
    failed_requests = [
        record
        for record in sorted_by_session
        if record.get("num_output_tokens", 0) == 0
    ]
    if not failed_requests:
        result.passed(f"no_errors [{check_label}]", "all requests produced output")
    else:
        result.fail(
            f"no_errors [{check_label}]",
            f"{len(failed_requests)} requests produced 0 output tokens",
        )

    # Interference requests should have output_tokens=1
    mismatched_interference_outputs = [
        record for record in interference_requests if record["num_output_tokens"] != 1
    ]
    if not mismatched_interference_outputs:
        result.passed(
            f"interference_output_tokens [{check_label}]",
            "all interference requests have 1 output token",
        )
    else:
        result.warn(
            f"interference_output_tokens [{check_label}]",
            f"{len(mismatched_interference_outputs)} interference requests had output != 1",
        )

    # Decode window overlap check on bench run
    decode_window_data = _load_decode_window_json(
        f"{output_dir}/{output_dir_tag}/bench"
    )
    if decode_window_data is not None:
        num_segments = decode_window_data.get("windows", {}).get(
            "num_selected_segments", 0
        )
        tbc_count = decode_window_data.get("tbc_in_window_stats", {}).get("count", 0)
        if num_segments == 0 or tbc_count == 0:
            result.fail(
                f"decode_window_overlap [{check_label}]",
                "no qualifying decode windows found — increase output tokens",
            )
        elif tbc_count < cfg.samples_per_length:
            result.warn(
                f"decode_window_overlap [{check_label}]",
                f"low sample count in decode window: {tbc_count} < {cfg.samples_per_length}",
            )
        else:
            result.passed(
                f"decode_window_overlap [{check_label}]",
                f"{tbc_count} samples in decode window",
            )
