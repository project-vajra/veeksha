#!/usr/bin/env python3
"""Generate a synthetic trace bundle for the bursty-timing experiment.

The bundle contains two trace variants for Veeksha's `claude_code` trace flavor:

1. `experiment2_probe_short.jsonl`
   The probe sessions use uniformly short waits.
2. `experiment2_probe_long.jsonl`
   The same probe sessions inject a few long waits after the shared prefix has
   grown large enough that a cache miss should be visible in TTFC.

The pressure sessions are identical in both files. This lets us attribute any
probe-session TTFC difference to wait structure rather than token or arrival
changes.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_LONG_GAPS = "8:2,12:10,18:30"


@dataclass(frozen=True)
class SessionRow:
    session_id: int
    turn_idx: int
    session_kind: str
    input_length: int
    new_input_length: int
    output_length: int
    wait_after_previous_response_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_idx": self.turn_idx,
            "session_kind": self.session_kind,
            "input_length": self.input_length,
            "new_input_length": self.new_input_length,
            "output_length": self.output_length,
            "wait_after_previous_response_s": self.wait_after_previous_response_s,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate short-gap and long-gap trace files for the bursty timing "
            "experiment."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("traces/bursty_timing"),
        help="Directory where the trace bundle will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the pressure sessions and probe template.",
    )
    parser.add_argument(
        "--warmup-pressure-sessions",
        type=int,
        default=48,
        help="Number of pressure sessions that arrive before the probe block.",
    )
    parser.add_argument(
        "--probe-sessions",
        type=int,
        default=8,
        help="Number of identical probe-session replicas to include.",
    )
    parser.add_argument(
        "--pressure-sessions-between-probes",
        type=int,
        default=48,
        help="Number of pressure sessions inserted between consecutive probe sessions.",
    )
    parser.add_argument(
        "--tail-pressure-sessions",
        type=int,
        default=480,
        help="Number of pressure sessions that arrive after the probe block.",
    )
    parser.add_argument(
        "--pressure-turns",
        type=int,
        default=16,
        help="Number of turns in each pressure session.",
    )
    parser.add_argument(
        "--probe-turns",
        type=int,
        default=24,
        help="Number of turns in each probe session.",
    )
    parser.add_argument(
        "--probe-root-input-tokens",
        type=int,
        default=8192,
        help="Prompt length of the probe session's first turn.",
    )
    parser.add_argument(
        "--probe-root-output-tokens",
        type=int,
        default=256,
        help="Output length of the probe session's first turn.",
    )
    parser.add_argument(
        "--probe-target-final-input-tokens",
        type=int,
        default=110000,
        help="Approximate total prompt length reached by the final probe turn.",
    )
    parser.add_argument(
        "--probe-short-wait-s",
        type=float,
        default=0.05,
        help="Short wait used for the probe sessions in the short-gap trace.",
    )
    parser.add_argument(
        "--probe-long-gaps",
        default=DEFAULT_LONG_GAPS,
        help=(
            "Comma-separated turn:seconds pairs for the long-gap variant. "
            "Turn indices are 0-based and refer to the turn dispatched after "
            "the wait, e.g. '12:2,20:10,28:30'."
        ),
    )
    parser.add_argument(
        "--recommended-arrival-interval-s",
        type=float,
        default=0.25,
        help="Recorded in metadata for the matching Veeksha sample configs.",
    )
    return parser.parse_args()


def parse_long_gap_spec(spec: str) -> dict[int, float]:
    parsed: dict[int, float] = {}
    if not spec:
        return parsed
    for chunk in spec.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        turn_text, gap_text = piece.split(":", maxsplit=1)
        turn_idx = int(turn_text)
        gap_s = float(gap_text)
        if turn_idx <= 0:
            raise ValueError(
                f"Long-gap turn indices must be positive; got {turn_idx}."
            )
        if gap_s <= 0:
            raise ValueError(f"Long gaps must be positive; got {gap_s}.")
        parsed[turn_idx] = gap_s
    return parsed


def bounded_round(value: float, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(value))))


def build_probe_template(
    num_turns: int,
    seed: int,
    *,
    root_input_tokens: int,
    root_output_tokens: int,
    target_final_input_tokens: int,
) -> list[dict[str, int]]:
    rng = random.Random(seed ^ 0x5F3759DF)
    if num_turns < 2:
        return [
            {
                "new_input_length": root_input_tokens,
                "output_length": root_output_tokens,
            }
        ]

    output_lengths = [root_output_tokens]
    for turn_idx in range(1, num_turns):
        base_output = 112 + 22 * (turn_idx % 4)
        output_jitter = rng.randint(-18, 26)
        output = bounded_round(
            base_output + output_jitter,
            minimum=72,
            maximum=224,
        )
        if turn_idx in {8, 12, 18, num_turns - 2}:
            output = bounded_round(
                output * 1.2,
                minimum=96,
                maximum=256,
            )
        output_lengths.append(output)

    carried_output_budget = sum(output_lengths[:-1])
    available_new_input_budget = max(
        0,
        target_final_input_tokens - root_input_tokens - carried_output_budget,
    )

    weights: list[float] = []
    for turn_idx in range(1, num_turns):
        progress = turn_idx / (num_turns - 1)
        base_weight = 1.0 + 1.1 * progress
        cycle_bonus = 0.15 * (turn_idx % 4)
        milestone_bonus = 0.6 if turn_idx in {8, 12, 18, num_turns - 2} else 0.0
        noise = rng.uniform(-0.08, 0.08)
        weights.append(max(0.35, base_weight + cycle_bonus + milestone_bonus + noise))

    weight_sum = sum(weights)
    target_new_inputs: list[int] = []
    remaining_budget = available_new_input_budget
    remaining_weight = weight_sum

    for index, weight in enumerate(weights):
        turns_left = len(weights) - index - 1
        min_remaining = turns_left * 768
        if remaining_weight <= 0:
            proposed = 768
        else:
            proposed = round(remaining_budget * (weight / remaining_weight))
        proposed = bounded_round(
            proposed,
            minimum=768,
            maximum=8192,
        )
        max_allowed = max(768, remaining_budget - min_remaining)
        proposed = min(proposed, max_allowed)
        target_new_inputs.append(proposed)
        remaining_budget -= proposed
        remaining_weight -= weight

    if remaining_budget != 0:
        target_new_inputs[-1] = bounded_round(
            target_new_inputs[-1] + remaining_budget,
            minimum=768,
            maximum=8192,
        )

    template = [
        {
            "new_input_length": root_input_tokens,
            "output_length": output_lengths[0],
        }
    ]
    for turn_idx, (new_input, output_length) in enumerate(
        zip(target_new_inputs, output_lengths[1:]),
        start=1,
    ):
        template.append(
            {
                "new_input_length": new_input,
                "output_length": output_length,
            }
        )
    return template


def sample_pressure_wait_s(rng: random.Random) -> float:
    bucket = rng.random()
    if bucket < 0.60:
        return round(rng.uniform(0.05, 0.25), 3)
    if bucket < 0.90:
        return round(rng.uniform(0.25, 1.50), 3)
    return round(rng.uniform(1.50, 6.00), 3)


def build_pressure_template(
    num_turns: int,
    rng: random.Random,
) -> list[dict[str, float | int]]:
    template: list[dict[str, float | int]] = []

    for turn_idx in range(num_turns):
        if turn_idx == 0:
            new_input = rng.randint(4096, 8192)
            output = rng.randint(160, 256)
            wait_s = 0.0
        else:
            roll = rng.random()
            if roll < 0.55:
                new_input = rng.randint(768, 2048)
                output = rng.randint(72, 144)
            elif roll < 0.90:
                new_input = rng.randint(2048, 4096)
                output = rng.randint(96, 192)
            else:
                new_input = rng.randint(4096, 8192)
                output = rng.randint(128, 256)

            if turn_idx % 4 == 0:
                new_input = bounded_round(
                    new_input * 1.25,
                    minimum=1024,
                    maximum=8192,
                )
                output = bounded_round(
                    output * 1.20,
                    minimum=96,
                    maximum=256,
                )
            wait_s = sample_pressure_wait_s(rng)

        template.append(
            {
                "new_input_length": new_input,
                "output_length": output,
                "wait_after_previous_response_s": wait_s,
            }
        )

    return template


def materialize_session_rows(
    *,
    session_id: int,
    session_kind: str,
    turns: Sequence[Mapping[str, float | int]],
    short_wait_s: float,
    long_gap_overrides: dict[int, float] | None = None,
) -> tuple[list[SessionRow], list[SessionRow]]:
    short_rows: list[SessionRow] = []
    long_rows: list[SessionRow] = []
    running_input_length = 0
    previous_output_length = 0

    for turn_idx, turn in enumerate(turns):
        new_input_length = int(turn["new_input_length"])
        output_length = int(turn["output_length"])

        if turn_idx == 0:
            running_input_length = new_input_length
            short_wait = 0.0
            long_wait = 0.0
        else:
            running_input_length += previous_output_length + new_input_length
            if session_kind == "probe":
                short_wait = short_wait_s
                long_wait = (
                    long_gap_overrides.get(turn_idx, short_wait_s)
                    if long_gap_overrides is not None
                    else short_wait_s
                )
            else:
                short_wait = float(turn["wait_after_previous_response_s"])
                long_wait = short_wait

        row_common = {
            "session_id": session_id,
            "turn_idx": turn_idx,
            "session_kind": session_kind,
            "input_length": running_input_length,
            "new_input_length": new_input_length,
            "output_length": output_length,
        }

        short_rows.append(
            SessionRow(
                **row_common,
                wait_after_previous_response_s=round(short_wait, 3),
            )
        )
        long_rows.append(
            SessionRow(
                **row_common,
                wait_after_previous_response_s=round(long_wait, 3),
            )
        )

        previous_output_length = output_length

    return short_rows, long_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def compute_running_input_lengths(turns: Sequence[Mapping[str, float | int]]) -> list[int]:
    running_input_lengths: list[int] = []
    running_input = 0
    previous_output = 0
    for turn_idx, turn in enumerate(turns):
        new_input = int(turn["new_input_length"])
        if turn_idx == 0:
            running_input = new_input
        else:
            running_input += previous_output + new_input
        running_input_lengths.append(running_input)
        previous_output = int(turn["output_length"])
    return running_input_lengths


def main() -> None:
    args = parse_args()
    long_gap_overrides = parse_long_gap_spec(args.probe_long_gaps)
    if any(turn_idx >= args.probe_turns for turn_idx in long_gap_overrides):
        raise ValueError(
            "All long-gap turn indices must be smaller than --probe-turns."
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_template = build_probe_template(
        args.probe_turns,
        args.seed,
        root_input_tokens=args.probe_root_input_tokens,
        root_output_tokens=args.probe_root_output_tokens,
        target_final_input_tokens=args.probe_target_final_input_tokens,
    )
    pressure_rng = random.Random(args.seed)
    probe_running_inputs = compute_running_input_lengths(probe_template)

    short_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    session_manifest: list[dict[str, Any]] = []

    session_plan: list[str] = []
    session_plan.extend(["pressure"] * args.warmup_pressure_sessions)
    for probe_idx in range(args.probe_sessions):
        session_plan.append("probe")
        if probe_idx < args.probe_sessions - 1:
            session_plan.extend(
                ["pressure"] * args.pressure_sessions_between_probes
            )
    session_plan.extend(["pressure"] * args.tail_pressure_sessions)

    total_sessions = len(session_plan)
    pressure_sessions = session_plan.count("pressure")
    total_requests = (
        pressure_sessions * args.pressure_turns + args.probe_sessions * args.probe_turns
    )
    arrival_horizon_s = round(
        max(0, total_sessions - 1) * args.recommended_arrival_interval_s,
        3,
    )
    probe_session_ids = [
        session_id for session_id, session_kind in enumerate(session_plan) if session_kind == "probe"
    ]

    for session_id, session_kind in enumerate(session_plan):
        is_probe = session_kind == "probe"

        if is_probe:
            turns: Sequence[Mapping[str, float | int]] = probe_template
            num_turns = args.probe_turns
        else:
            turns = build_pressure_template(args.pressure_turns, pressure_rng)
            num_turns = args.pressure_turns

        short_session, long_session = materialize_session_rows(
            session_id=session_id,
            session_kind=session_kind,
            turns=turns,
            short_wait_s=args.probe_short_wait_s,
            long_gap_overrides=long_gap_overrides,
        )
        short_rows.extend(row.to_dict() for row in short_session)
        long_rows.extend(row.to_dict() for row in long_session)
        session_manifest.append(
            {
                "original_session_id": session_id,
                "expected_runtime_session_id": session_id,
                "session_kind": session_kind,
                "num_turns": num_turns,
            }
        )

    short_trace_path = output_dir / "experiment2_probe_short.jsonl"
    long_trace_path = output_dir / "experiment2_probe_long.jsonl"
    metadata_path = output_dir / "experiment2_metadata.json"
    session_manifest_path = output_dir / "experiment2_session_manifest.jsonl"

    write_jsonl(short_trace_path, short_rows)
    write_jsonl(long_trace_path, long_rows)
    write_jsonl(session_manifest_path, session_manifest)

    metadata = {
        "trace_family": "experiment2_bursty_timing",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "files": {
            "short_trace": short_trace_path.as_posix(),
            "long_trace": long_trace_path.as_posix(),
            "session_manifest": session_manifest_path.as_posix(),
        },
        "layout": {
            "warmup_pressure_sessions": args.warmup_pressure_sessions,
            "probe_sessions": args.probe_sessions,
            "pressure_sessions_between_probes": args.pressure_sessions_between_probes,
            "tail_pressure_sessions": args.tail_pressure_sessions,
            "total_sessions": total_sessions,
            "pressure_sessions": pressure_sessions,
            "pressure_turns": args.pressure_turns,
            "probe_turns": args.probe_turns,
            "total_requests_per_trace": total_requests,
        },
        "probe": {
            "short_wait_s": args.probe_short_wait_s,
            "long_gap_turns_s": long_gap_overrides,
            "root_input_tokens": args.probe_root_input_tokens,
            "target_final_input_tokens": args.probe_target_final_input_tokens,
            "reference_running_input_lengths": {
                str(turn_idx): probe_running_inputs[turn_idx]
                for turn_idx in range(len(probe_running_inputs))
            },
            "original_session_ids": probe_session_ids,
            "expected_runtime_session_ids": probe_session_ids,
        },
        "recommended_benchmark": {
            "traffic_scheduler": {
                "type": "rate",
                "interval_generator": {
                    "type": "fixed",
                    "interval": args.recommended_arrival_interval_s,
                },
                "estimated_arrival_horizon_s": arrival_horizon_s,
            },
            "runtime": {
                "max_sessions": total_sessions,
                "pregenerate_sessions": True,
            },
        },
        "notes": [
            "Runtime session IDs match the original session_id values because the trace generator groups sessions by ascending session_id and wrap_mode is expected to be false.",
            "Probe sessions are identical between trace variants except for wait_after_previous_response_s.",
            "Pressure sessions are identical between trace variants.",
            "Pressure sessions are intentionally large and long-lived so they keep KV pressure high across the probe session's long waits.",
            "Probe sessions are interleaved with pressure sessions to avoid synchronized idle gaps across all probe replicas.",
        ],
    }

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote trace bundle to {output_dir}")
    print(f"  short trace: {short_trace_path}")
    print(f"  long trace:  {long_trace_path}")
    print(f"  metadata:    {metadata_path}")


if __name__ == "__main__":
    main()
