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
from veeksha.microbench.expand import _prefill_iters

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
        return all(s != "FAIL" for s, _, _ in self.checks)

    def summary(self) -> str:
        lines = []
        for status, name, detail in self.checks:
            tag = {"PASS": "  PASS", "WARN": "  WARN", "FAIL": "**FAIL"}[status]
            line = f"{tag}  {name}"
            if detail:
                line += f" — {detail}"
            lines.append(line)
        n_pass = sum(1 for s, _, _ in self.checks if s == "PASS")
        n_warn = sum(1 for s, _, _ in self.checks if s == "WARN")
        n_fail = sum(1 for s, _, _ in self.checks if s == "FAIL")
        lines.append(f"\n{n_pass} passed, {n_warn} warnings, {n_fail} failures")
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
        r = ValidationResult()
        r.fail("unknown_type", f"No validator for type '{cfg.type}'")
        return r


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


def _validate_prefill(cfg: MicrobenchmarkConfig, output_dir: str) -> ValidationResult:
    v = ValidationResult()
    metrics = _load_request_metrics(output_dir)
    if metrics is None:
        v.fail("metrics_found", "No request_level_metrics.jsonl found")
        return v
    v.passed("metrics_found")

    expected_count = len(cfg.input_lengths) * cfg.samples_per_length

    # Session count
    if len(metrics) == expected_count:
        v.passed("session_count", f"{len(metrics)} requests")
    else:
        v.fail("session_count", f"expected {expected_count}, got {len(metrics)}")

    # No errors
    errors = [r for r in metrics if r.get("num_output_tokens", 0) == 0]
    if not errors:
        v.passed("no_errors", "all requests produced output")
    else:
        v.fail("no_errors", f"{len(errors)} requests produced 0 output tokens")

    # Output tokens = expected
    bad_out = [r for r in metrics if r["num_output_tokens"] != cfg.output_tokens]
    if not bad_out:
        v.passed(
            "output_tokens", f"all requests produced {cfg.output_tokens} output tokens"
        )
    else:
        v.warn(
            "output_tokens",
            f"{len(bad_out)} requests had unexpected output token count",
        )

    # Sequential execution (concurrent=1): each request dispatched after prior completed
    by_session = sorted(metrics, key=lambda r: r["session_id"])
    sequential = True
    for i in range(1, len(by_session)):
        prev_comp = by_session[i - 1]["client_completed_at"]
        curr_disp = by_session[i]["scheduler_dispatched_at"]
        if curr_disp < prev_comp - 0.01:  # 10ms tolerance
            sequential = False
            break
    if sequential:
        v.passed("sequential_execution", "requests executed one at a time")
    else:
        v.warn("sequential_execution", "some requests overlapped (concurrent != 1?)")

    # Prompt tokens match stair pattern
    for i, r in enumerate(by_session):
        length_idx = i // cfg.samples_per_length
        if length_idx < len(cfg.input_lengths):
            expected_prompt = cfg.input_lengths[length_idx]
            actual_prompt = r["target_num_delta_prompt_tokens"]
            if actual_prompt != expected_prompt:
                v.warn(
                    "prompt_tokens_stair",
                    f"session {r['session_id']}: expected {expected_prompt}, got {actual_prompt}",
                )
                break
    else:
        v.passed("prompt_tokens_stair", "prompt tokens follow expected stair pattern")

    return v


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def _validate_decode(cfg: MicrobenchmarkConfig, output_dir: str) -> ValidationResult:
    v = ValidationResult()

    # Validate each (batch_size, input_length) run
    for batch_size in cfg.batch_sizes:
        for input_length in cfg.input_lengths:
            tag = f"bs={batch_size},il={input_length}"
            _validate_one_decode_run(v, cfg, batch_size, input_length, output_dir, tag)

    return v


def _validate_one_decode_run(
    v: ValidationResult,
    cfg: MicrobenchmarkConfig,
    batch_size: int,
    input_length: int,
    output_dir: str,
    tag: str,
) -> None:
    # We can't easily map which sub-run dir belongs to which (bs, il) pair
    # from the output_dir alone, so we validate the sweep-level metrics.
    # For now, check the overall sweep directory.
    metrics = _load_request_metrics(f"{output_dir}/bs={batch_size}_il={input_length}")
    if metrics is None:
        v.fail(
            f"metrics_found [{tag}]",
            "No request_level_metrics.jsonl found in decode dir",
        )
        return

    # Filter to requests matching this input_length
    matching = [
        r for r in metrics if r["target_num_delta_prompt_tokens"] == input_length
    ]

    if not matching:
        v.warn(f"matching_requests [{tag}]", f"no requests with prompt={input_length}")
        return
    v.passed(f"matching_requests [{tag}]", f"{len(matching)} requests")

    # FCFS ordering: first tokens should arrive roughly in session order
    by_session = sorted(matching, key=lambda r: r["session_id"])
    first_tokens = [r["client_picked_up_at"] + r["ttfc"] for r in by_session]
    out_of_order = sum(
        1
        for i in range(1, len(first_tokens))
        if first_tokens[i] < first_tokens[i - 1] - 0.005
    )
    if out_of_order == 0:
        v.passed(f"fcfs_order [{tag}]", "first tokens arrived in session order")
    else:
        v.warn(
            f"fcfs_order [{tag}]",
            f"{out_of_order} out-of-order first tokens (engine may batch prefills)",
        )

    # Decode window overlap check
    dw = _load_decode_window_json(f"{output_dir}/bs={batch_size}_il={input_length}")
    if dw is not None:
        num_segments = dw.get("windows", {}).get("num_selected_segments", 0)
        tbc_count = dw.get("tbc_in_window_stats", {}).get("count", 0)
        if num_segments == 0 or tbc_count == 0:
            v.fail(
                f"decode_window_overlap [{tag}]",
                "no qualifying decode windows found — increase output tokens",
            )
        elif tbc_count < cfg.samples_per_length:
            v.warn(
                f"decode_window_overlap [{tag}]",
                f"low sample count in decode window: {tbc_count} < {cfg.samples_per_length}",
            )
        else:
            v.passed(
                f"decode_window_overlap [{tag}]",
                f"{tbc_count} samples in decode window",
            )


# ---------------------------------------------------------------------------
# Mixed batch
# ---------------------------------------------------------------------------


def _validate_mixed(cfg: MicrobenchmarkConfig, output_dir: str) -> ValidationResult:
    v = ValidationResult()

    for batch_size in cfg.batch_sizes:
        for decode_input_length in cfg.decode_input_lengths:
            for prefill_kv_length in cfg.prefill_kv_lengths:
                for incremental_prefill_size in cfg.incremental_prefill_sizes:
                    tag = f"bs={batch_size},dil={decode_input_length},kv={prefill_kv_length},dp={incremental_prefill_size}"
                    dir_tag = f"bs={batch_size}_dil={decode_input_length}_kv={prefill_kv_length}_dp={incremental_prefill_size}"
                    _validate_one_mixed_run(
                        v,
                        cfg,
                        batch_size,
                        decode_input_length,
                        prefill_kv_length,
                        incremental_prefill_size,
                        output_dir,
                        dir_tag,
                        tag,
                    )

    return v


def _validate_one_mixed_run(
    v: ValidationResult,
    cfg: MicrobenchmarkConfig,
    batch_size: int,
    decode_input_length: int,
    prefill_kv_length: int,
    incremental_prefill_size: int,
    output_dir: str,
    dir_tag: str,
    tag: str,
) -> None:
    # -- Warmup validation --
    warmup_metrics = _load_request_metrics(f"{output_dir}/{dir_tag}/warmup")
    if warmup_metrics is None:
        v.warn(f"warmup_found [{tag}]", "no warmup metrics found")
    else:
        v.passed(f"warmup_found [{tag}]", f"{len(warmup_metrics)} warmup requests")

    # -- Benchmark validation --
    metrics = _load_request_metrics(f"{output_dir}/{dir_tag}/bench")
    if metrics is None:
        v.fail(f"metrics_found [{tag}]", "No request_level_metrics.jsonl found")
        return
    v.passed(f"metrics_found [{tag}]")

    by_session = sorted(metrics, key=lambda r: r["session_id"])

    # Identify decode vs interference by output token count
    samples_per_prefill = _prefill_iters(
        incremental_prefill_size,
        cfg.engine_chunk_size,
        batch_size,
    )
    num_prefill_requests = math.ceil(cfg.samples_per_length / samples_per_prefill)
    expected_bench_sessions = batch_size + num_prefill_requests

    if len(by_session) != expected_bench_sessions:
        v.warn(
            f"session_count [{tag}]",
            f"expected {expected_bench_sessions}, got {len(by_session)}",
        )
    else:
        v.passed(f"session_count [{tag}]", f"{len(by_session)} sessions")

    # Split into decode and interference requests based on session order
    decode_reqs = by_session[:batch_size]
    interference_reqs = by_session[batch_size:]

    # Check decode request prompt tokens
    bad_decode_prompts = [
        r
        for r in decode_reqs
        if r["target_num_delta_prompt_tokens"] != decode_input_length
    ]
    if not bad_decode_prompts:
        v.passed(
            f"decode_prompts [{tag}]",
            f"all decode requests have prompt={decode_input_length}",
        )
    else:
        v.warn(
            f"decode_prompts [{tag}]",
            f"{len(bad_decode_prompts)} decode requests had wrong prompt token count",
        )

    # Check interference request prompt tokens
    expected_interf_prompt = prefill_kv_length + incremental_prefill_size
    bad_interf_prompts = [
        r
        for r in interference_reqs
        if r["target_num_delta_prompt_tokens"] != expected_interf_prompt
    ]
    if not bad_interf_prompts:
        v.passed(
            f"interference_prompts [{tag}]",
            f"all interference requests have prompt={expected_interf_prompt}",
        )
    else:
        v.warn(
            f"interference_prompts [{tag}]",
            f"{len(bad_interf_prompts)} interference requests had wrong prompt token count",
        )

    # FCFS: decode first tokens before interference first tokens
    if decode_reqs and interference_reqs:
        decode_first_tokens = [
            r["client_picked_up_at"] + r["ttfc"] for r in decode_reqs
        ]
        interf_first_tokens = [
            r["client_picked_up_at"] + r["ttfc"] for r in interference_reqs
        ]
        max_decode_ft = max(decode_first_tokens)
        min_interf_ft = min(interf_first_tokens)

        if max_decode_ft < min_interf_ft:
            v.passed(
                f"fcfs_decode_before_interference [{tag}]",
                f"decode last_ft={max_decode_ft:.3f} < interference first_ft={min_interf_ft:.3f}",
            )
        else:
            v.fail(
                f"fcfs_decode_before_interference [{tag}]",
                f"decode last_ft={max_decode_ft:.3f} >= interference first_ft={min_interf_ft:.3f} "
                f"— interference started before all decodes entered decode phase",
            )

    # No errors
    errors = [r for r in by_session if r.get("num_output_tokens", 0) == 0]
    if not errors:
        v.passed(f"no_errors [{tag}]", "all requests produced output")
    else:
        v.fail(f"no_errors [{tag}]", f"{len(errors)} requests produced 0 output tokens")

    # Interference requests should have output_tokens=1
    bad_interf_out = [r for r in interference_reqs if r["num_output_tokens"] != 1]
    if not bad_interf_out:
        v.passed(
            f"interference_output_tokens [{tag}]",
            "all interference requests have 1 output token",
        )
    else:
        v.warn(
            f"interference_output_tokens [{tag}]",
            f"{len(bad_interf_out)} interference requests had output != 1",
        )

    # Decode window overlap check on bench run
    dw = _load_decode_window_json(f"{output_dir}/{dir_tag}/bench")
    if dw is not None:
        num_segments = dw.get("windows", {}).get("num_selected_segments", 0)
        tbc_count = dw.get("tbc_in_window_stats", {}).get("count", 0)
        if num_segments == 0 or tbc_count == 0:
            v.fail(
                f"decode_window_overlap [{tag}]",
                "no qualifying decode windows found — increase output tokens",
            )
        elif tbc_count < cfg.samples_per_length:
            v.warn(
                f"decode_window_overlap [{tag}]",
                f"low sample count in decode window: {tbc_count} < {cfg.samples_per_length}",
            )
        else:
            v.passed(
                f"decode_window_overlap [{tag}]",
                f"{tbc_count} samples in decode window",
            )
