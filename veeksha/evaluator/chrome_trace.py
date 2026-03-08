"""Chrome Trace Event Format output for timeline visualization.

Generates JSON files loadable in chrome://tracing or ui.perfetto.dev
to inspect dispatch ordering, interference overlap, and per-token timing.
"""

import json
from typing import Any, Dict, List


def generate_chrome_trace(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Convert request-level metric rows to Chrome Trace Event Format.

    Args:
        rows: Row dicts as produced by TextPerformanceEvaluator._export_request_rows().
        output_path: File path for the output JSON trace.
    """
    if not rows:
        with open(output_path, "w") as f:
            json.dump([], f)
        return

    # Compute epoch: earliest scheduler_dispatched_at (seconds) -> us offset base
    epoch = min(r["scheduler_dispatched_at"] for r in rows)

    def to_us(ts: float) -> float:
        """Convert absolute timestamp (seconds) to us relative to epoch."""
        return (ts - epoch) * 1_000_000

    events: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        request_id = row["request_id"]
        session_id = row["session_id"]
        num_input = row["num_delta_prompt_tokens"]
        num_output = row["num_output_tokens"]
        tid = f"R{idx} / {session_id} / {num_input}\u2192{num_output}"
        pid = 1

        pickup_us = to_us(row["client_picked_up_at"])
        completed_us = to_us(row["client_completed_at"])
        ttfc_us = row["ttfc"] * 1_000_000
        tbc = row.get("tbc", [])

        common_args = {
            "request_id": request_id,
            "session_id": session_id,
            "num_delta_prompt_tokens": num_input,
            "num_total_prompt_tokens": row["num_total_prompt_tokens"],
            "num_output_tokens": num_output,
            "num_total_tokens": row["num_total_tokens"],
            "ttfc": row["ttfc"],
            "tpot": row["tpot"],
            "end_to_end_latency": row["end_to_end_latency"],
            "output_throughput": row["output_throughput"],
        }

        # Prefill phase: pickup -> pickup + ttfc (always present)
        events.append(
            {
                "name": "Prefill",
                "cat": "prefill",
                "ph": "X",
                "ts": pickup_us,
                "dur": ttfc_us,
                "pid": pid,
                "tid": tid,
                "args": common_args,
            }
        )

        # Decode phase: individual token events from tbc (time-between-completions)
        if num_output > 1 and tbc:
            token_start = pickup_us + ttfc_us
            for token_idx, token_dur_s in enumerate(tbc):
                token_dur_us = token_dur_s * 1_000_000
                events.append(
                    {
                        "name": f"T{token_idx + 1}",
                        "cat": "decode",
                        "ph": "X",
                        "ts": token_start,
                        "dur": token_dur_us,
                        "pid": pid,
                        "tid": tid,
                        "args": {"token_index": token_idx + 1, "tbc_s": token_dur_s},
                    }
                )
                token_start += token_dur_us
        elif num_output == 1:
            # Single output token: the ttfc already covers it, nothing more
            pass
        else:
            # Fallback: one decode block if tbc is missing but there are output tokens
            decode_start = pickup_us + ttfc_us
            decode_dur = completed_us - decode_start
            if decode_dur > 0:
                events.append(
                    {
                        "name": "Decode",
                        "cat": "decode",
                        "ph": "X",
                        "ts": decode_start,
                        "dur": decode_dur,
                        "pid": pid,
                        "tid": tid,
                        "args": common_args,
                    }
                )

    with open(output_path, "w") as f:
        json.dump(events, f)
