#!/usr/bin/env python3
"""Generate matched-volume traces for the workload-shape case study.

The output is tailored to Veeksha's ``timed_synthetic_session`` trace flavor.

Workload A:
- 30 sessions
- 5 requests per session
- Linear history chain

Workload B:
- 10 sessions
- 15 requests per session
- Fixed DAG topology matching the case-study description

Both workloads use the same fresh-token budget by default:
- 150 total requests
- 500 fresh input tokens per request
- 300 output tokens per request

``input_length`` is derived from the chosen history-parent lineage because
``timed_synthetic_session`` only inherits prompt history from the edge marked
as ``history_parent``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LINEAR_TRACE_NAME = "workload_a_linear.jsonl"
DAG_TRACE_NAME = "workload_b_dag.jsonl"
METADATA_NAME = "workload_shape_metadata.json"

DAG_PARENT_MAP: dict[int, list[int]] = {
    0: [],
    1: [0],
    2: [1],
    3: [1],
    4: [1],
    5: [2],
    6: [2],
    7: [2],
    8: [3],
    9: [3],
    10: [3],
    11: [4],
    12: [4],
    13: [4],
    14: [1, 7, 10, 13],
}


@dataclass(frozen=True)
class NodeSpec:
    node_id: int
    parent_nodes: list[int]
    history_parent: int | None
    wait_after_ready: float

    def to_session_context(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_nodes": list(self.parent_nodes),
            "history_parent": self.history_parent,
            "wait_after_ready": self.wait_after_ready,
        }


@dataclass(frozen=True)
class TraceRow:
    session_id: int
    node_id: int
    workload: str
    input_length: int
    new_input_length: int
    output_length: int
    cacheable_history_tokens: int
    session_context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "node_id": self.node_id,
            "workload": self.workload,
            "input_length": self.input_length,
            "new_input_length": self.new_input_length,
            "output_length": self.output_length,
            "cacheable_history_tokens": self.cacheable_history_tokens,
            "session_context": self.session_context,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate matched-volume linear and DAG traces for Veeksha's "
            "timed_synthetic_session trace flavor."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("traces/workload_shape"),
        help="Directory where the generated traces and metadata will be written.",
    )
    parser.add_argument(
        "--linear-sessions",
        type=int,
        default=30,
        help="Number of sessions in workload A.",
    )
    parser.add_argument(
        "--linear-requests",
        type=int,
        default=5,
        help="Number of requests per session in workload A.",
    )
    parser.add_argument(
        "--dag-sessions",
        type=int,
        default=10,
        help="Number of sessions in workload B.",
    )
    parser.add_argument(
        "--fresh-input-tokens",
        type=int,
        default=500,
        help="Fresh input tokens per request in both workloads.",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=300,
        help="Output tokens per request in both workloads.",
    )
    parser.add_argument(
        "--wait-after-ready-s",
        type=float,
        default=0.0,
        help=(
            "Wait inserted on non-root nodes. Roots stay at 0.0 so session "
            "arrivals are controlled by the traffic scheduler."
        ),
    )
    parser.add_argument(
        "--dag-final-history-parent",
        type=int,
        default=13,
        help=(
            "History parent for DAG node 14. Must be one of 1, 7, 10, or 13 "
            "to match the documented DAG join."
        ),
    )
    return parser.parse_args()


def topological_sort(parent_map: Mapping[int, Sequence[int]]) -> list[int]:
    children: dict[int, list[int]] = {node_id: [] for node_id in parent_map}
    incoming_counts = {node_id: len(parents) for node_id, parents in parent_map.items()}
    for node_id, parents in parent_map.items():
        for parent in parents:
            children.setdefault(parent, []).append(node_id)

    ready = sorted(node_id for node_id, count in incoming_counts.items() if count == 0)
    order: list[int] = []

    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for child in sorted(children.get(node_id, [])):
            incoming_counts[child] -= 1
            if incoming_counts[child] == 0:
                ready.append(child)
                ready.sort()

    if len(order) != len(parent_map):
        raise ValueError("Topology contains a cycle or references missing nodes.")
    return order


def build_linear_template(
    *,
    num_requests: int,
    wait_after_ready_s: float,
) -> list[NodeSpec]:
    template: list[NodeSpec] = []
    for node_id in range(num_requests):
        if node_id == 0:
            template.append(
                NodeSpec(
                    node_id=node_id,
                    parent_nodes=[],
                    history_parent=None,
                    wait_after_ready=0.0,
                )
            )
            continue
        template.append(
            NodeSpec(
                node_id=node_id,
                parent_nodes=[node_id - 1],
                history_parent=node_id - 1,
                wait_after_ready=wait_after_ready_s,
            )
        )
    return template


def build_dag_template(
    *,
    wait_after_ready_s: float,
    final_history_parent: int,
) -> list[NodeSpec]:
    final_parents = DAG_PARENT_MAP[14]
    if final_history_parent not in final_parents:
        raise ValueError(
            "--dag-final-history-parent must be one of "
            f"{', '.join(str(parent) for parent in final_parents)}."
        )

    template: list[NodeSpec] = []
    for node_id in topological_sort(DAG_PARENT_MAP):
        parent_nodes = list(DAG_PARENT_MAP[node_id])
        if not parent_nodes:
            history_parent = None
            wait_after_ready = 0.0
        elif node_id == 14:
            history_parent = final_history_parent
            wait_after_ready = wait_after_ready_s
        else:
            history_parent = parent_nodes[0]
            wait_after_ready = wait_after_ready_s

        template.append(
            NodeSpec(
                node_id=node_id,
                parent_nodes=parent_nodes,
                history_parent=history_parent,
                wait_after_ready=wait_after_ready,
            )
        )
    return template


def materialize_trace_rows(
    *,
    workload: str,
    num_sessions: int,
    template: Sequence[NodeSpec],
    fresh_input_tokens: int,
    output_tokens: int,
) -> list[TraceRow]:
    input_lengths_by_node: dict[int, int] = {}
    output_lengths_by_node = {node.node_id: output_tokens for node in template}

    for node in template:
        if node.history_parent is None:
            cacheable_history_tokens = 0
        else:
            parent_prompt_tokens = input_lengths_by_node[node.history_parent]
            parent_output_tokens = output_lengths_by_node[node.history_parent]
            cacheable_history_tokens = parent_prompt_tokens + parent_output_tokens
        input_lengths_by_node[node.node_id] = (
            fresh_input_tokens + cacheable_history_tokens
        )

    rows: list[TraceRow] = []
    for session_id in range(num_sessions):
        for node in template:
            input_length = input_lengths_by_node[node.node_id]
            rows.append(
                TraceRow(
                    session_id=session_id,
                    node_id=node.node_id,
                    workload=workload,
                    input_length=input_length,
                    new_input_length=fresh_input_tokens,
                    output_length=output_tokens,
                    cacheable_history_tokens=input_length - fresh_input_tokens,
                    session_context=node.to_session_context(),
                )
            )
    return rows


def summarize_trace(rows: Sequence[TraceRow]) -> dict[str, Any]:
    session_ids = {row.session_id for row in rows}
    unique_node_ids = sorted({row.node_id for row in rows})
    total_requests = len(rows)
    total_new_input_tokens = sum(row.new_input_length for row in rows)
    total_output_tokens = sum(row.output_length for row in rows)
    total_effective_input_tokens = sum(row.input_length for row in rows)
    total_cacheable_history_tokens = sum(row.cacheable_history_tokens for row in rows)
    average_effective_input_tokens = (
        total_effective_input_tokens / total_requests if total_requests else 0.0
    )

    return {
        "sessions": len(session_ids),
        "requests_per_session": len(unique_node_ids),
        "total_requests": total_requests,
        "total_new_input_tokens": total_new_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_effective_input_tokens": total_effective_input_tokens,
        "total_cacheable_history_tokens": total_cacheable_history_tokens,
        "average_effective_input_tokens": round(average_effective_input_tokens, 3),
        "max_effective_input_tokens": max((row.input_length for row in rows), default=0),
        "min_effective_input_tokens": min((row.input_length for row in rows), default=0),
        "node_input_lengths": {
            str(node_id): rows_for_node[0].input_length
            for node_id, rows_for_node in group_rows_by_node(rows).items()
        },
    }


def group_rows_by_node(rows: Sequence[TraceRow]) -> dict[int, list[TraceRow]]:
    grouped: dict[int, list[TraceRow]] = {}
    for row in rows:
        grouped.setdefault(row.node_id, []).append(row)
    return grouped


def validate_matched_fresh_volume(
    linear_summary: Mapping[str, Any],
    dag_summary: Mapping[str, Any],
) -> None:
    comparable_keys = (
        "total_requests",
        "total_new_input_tokens",
        "total_output_tokens",
    )
    mismatches = [
        key
        for key in comparable_keys
        if linear_summary[key] != dag_summary[key]
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: linear={linear_summary[key]} dag={dag_summary[key]}"
            for key in mismatches
        )
        raise ValueError(f"Fresh-token volumes do not match: {details}")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True))
            handle.write("\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    linear_template = build_linear_template(
        num_requests=args.linear_requests,
        wait_after_ready_s=args.wait_after_ready_s,
    )
    dag_template = build_dag_template(
        wait_after_ready_s=args.wait_after_ready_s,
        final_history_parent=args.dag_final_history_parent,
    )

    linear_rows = materialize_trace_rows(
        workload="linear",
        num_sessions=args.linear_sessions,
        template=linear_template,
        fresh_input_tokens=args.fresh_input_tokens,
        output_tokens=args.output_tokens,
    )
    dag_rows = materialize_trace_rows(
        workload="dag",
        num_sessions=args.dag_sessions,
        template=dag_template,
        fresh_input_tokens=args.fresh_input_tokens,
        output_tokens=args.output_tokens,
    )

    linear_summary = summarize_trace(linear_rows)
    dag_summary = summarize_trace(dag_rows)
    validate_matched_fresh_volume(linear_summary, dag_summary)

    linear_path = output_dir / LINEAR_TRACE_NAME
    dag_path = output_dir / DAG_TRACE_NAME
    metadata_path = output_dir / METADATA_NAME

    write_jsonl(linear_path, (row.to_dict() for row in linear_rows))
    write_jsonl(dag_path, (row.to_dict() for row in dag_rows))

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_flavor": "timed_synthetic_session",
        "files": {
            "linear_trace": linear_path.as_posix(),
            "dag_trace": dag_path.as_posix(),
        },
        "parameters": {
            "linear_sessions": args.linear_sessions,
            "linear_requests": args.linear_requests,
            "dag_sessions": args.dag_sessions,
            "dag_requests": len(dag_template),
            "fresh_input_tokens": args.fresh_input_tokens,
            "output_tokens": args.output_tokens,
            "wait_after_ready_s": args.wait_after_ready_s,
            "dag_final_history_parent": args.dag_final_history_parent,
        },
        "topologies": {
            "linear": [node.to_session_context() for node in linear_template],
            "dag": [node.to_session_context() for node in dag_template],
        },
        "workloads": {
            "linear": linear_summary,
            "dag": dag_summary,
        },
        "comparison": {
            "matched_total_requests": (
                linear_summary["total_requests"] == dag_summary["total_requests"]
            ),
            "matched_total_new_input_tokens": (
                linear_summary["total_new_input_tokens"]
                == dag_summary["total_new_input_tokens"]
            ),
            "matched_total_output_tokens": (
                linear_summary["total_output_tokens"]
                == dag_summary["total_output_tokens"]
            ),
            "effective_input_ratio_dag_to_linear": round(
                dag_summary["total_effective_input_tokens"]
                / linear_summary["total_effective_input_tokens"],
                6,
            ),
            "cacheable_history_ratio_dag_to_linear": round(
                dag_summary["total_cacheable_history_tokens"]
                / linear_summary["total_cacheable_history_tokens"],
                6,
            ),
        },
    }
    write_json(metadata_path, metadata)

    print(f"Wrote linear trace: {linear_path}")
    print(f"Wrote DAG trace:    {dag_path}")
    print(f"Wrote metadata:     {metadata_path}")
    print(
        "Matched fresh volume: "
        f"{linear_summary['total_requests']} requests, "
        f"{linear_summary['total_new_input_tokens']} new input tokens, "
        f"{linear_summary['total_output_tokens']} output tokens"
    )
    print(
        "Effective input totals: "
        f"linear={linear_summary['total_effective_input_tokens']}, "
        f"dag={dag_summary['total_effective_input_tokens']}"
    )
    print(
        "Cacheable history totals: "
        f"linear={linear_summary['total_cacheable_history_tokens']}, "
        f"dag={dag_summary['total_cacheable_history_tokens']}"
    )


if __name__ == "__main__":
    main()
