"""Audits upstream lm-eval task dataset loadability under the current `datasets` version.

This repository vendors a fork of the lm-eval harness and also ships a compatibility
shim at `lm_eval/` (repo root). That shim can shadow the upstream pip package.

For the purpose of auditing task availability under `datasets==4.x`, this script
intentionally imports the **pip-installed** `lm_eval` by changing the working
directory to `/tmp` before importing. This prevents the repo-local shim from
being discovered via the current working directory.

The audit is intentionally lightweight: it does **not** download full datasets.
Instead, it calls `datasets.load_dataset_builder()` to validate that the dataset
builder can be constructed for each unique (dataset_path, dataset_name, kwargs)
pair referenced by tasks.

Outputs:
  - JSON report with per-builder results and impacted task names.
  - Markdown summary with high-signal counts and the largest failure clusters.

Example:
  python scripts/audit_lmeval_tasks_datasets4.py \
    --output-dir analysis/lmeval_datasets4_audit
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BuilderKey:
    dataset_path: str
    dataset_name: str | None
    dataset_kwargs: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "dataset_kwargs": self.dataset_kwargs,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_repo_root() / "analysis" / "lmeval_datasets4_audit"),
        help="Directory to write audit artifacts (JSON + Markdown).",
    )
    parser.add_argument(
        "--max-builders",
        type=int,
        default=0,
        help="If >0, limit the number of unique dataset builder probes.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first dataset builder failure (debugging).",
    )
    return parser.parse_args()


def _sanitize_for_json(x: Any) -> Any:
    """Best-effort conversion to JSON-serializable types."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_sanitize_for_json(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _sanitize_for_json(v) for k, v in x.items()}
    return str(x)


def _iter_tasks() -> Iterable[tuple[str, dict[str, Any]]]:
    # IMPORTANT: avoid importing the repo-local `lm_eval` shim.
    os.chdir("/tmp")

    from lm_eval.tasks import TaskManager  # type: ignore[import-not-found]

    tm = TaskManager()
    for task_name, meta in tm.task_index.items():
        if meta.get("type") in {"task", "python_task"}:
            yield task_name, meta


def _load_task_yaml(yaml_path: str) -> dict[str, Any]:
    # IMPORTANT: avoid importing the repo-local `lm_eval` shim.
    os.chdir("/tmp")

    from lm_eval.utils import load_yaml_config  # type: ignore[import-not-found]

    return load_yaml_config(yaml_path)


def _builder_supported_kwargs() -> set[str]:
    import datasets

    sig = inspect.signature(datasets.load_dataset_builder)
    # Exclude mandatory `path` and `name` which we pass positionally.
    return {k for k in sig.parameters.keys() if k not in {"path", "name"}}


def _compute_builder_key(task_yaml: dict[str, Any]) -> BuilderKey | None:
    dataset_path = task_yaml.get("dataset_path")
    if not dataset_path:
        return None
    dataset_name = task_yaml.get("dataset_name")
    dataset_kwargs = task_yaml.get("dataset_kwargs") or {}
    if not isinstance(dataset_kwargs, dict):
        dataset_kwargs = {"_non_dict_dataset_kwargs": str(dataset_kwargs)}
    dataset_kwargs = _sanitize_for_json(dataset_kwargs)
    return BuilderKey(
        dataset_path=str(dataset_path),
        dataset_name=None if dataset_name in {None, "null"} else str(dataset_name),
        dataset_kwargs=dataset_kwargs,
    )


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reduce progress bar spam from `datasets`.
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Snapshot environment versions (import from /tmp to avoid shadowing).
    os.chdir("/tmp")
    import datasets
    import lm_eval

    env = {
        "python": sys.version,
        "datasets_version": datasets.__version__,
        "lm_eval_version": getattr(lm_eval, "__version__", "unknown"),
    }

    supported_kwargs = _builder_supported_kwargs()

    task_to_key: dict[str, BuilderKey | None] = {}
    key_to_tasks: dict[str, list[str]] = defaultdict(list)

    yaml_parse_errors: dict[str, str] = {}
    tasks_missing_dataset_path: list[str] = []

    for task_name, meta in _iter_tasks():
        yaml_path = meta.get("yaml_path")
        if not yaml_path:
            yaml_parse_errors[task_name] = "missing_yaml_path"
            continue
        try:
            task_yaml = _load_task_yaml(yaml_path)
        except Exception as e:  # noqa: BLE001
            yaml_parse_errors[task_name] = f"{type(e).__name__}: {e}"
            continue
        key = _compute_builder_key(task_yaml)
        task_to_key[task_name] = key
        if key is None:
            tasks_missing_dataset_path.append(task_name)
            continue
        key_str = json.dumps(key.to_jsonable(), sort_keys=True)
        key_to_tasks[key_str].append(task_name)

    # Probe dataset builders.
    builder_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    keys = list(key_to_tasks.keys())
    if args.max_builders and args.max_builders > 0:
        keys = keys[: args.max_builders]

    for idx, key_str in enumerate(keys, start=1):
        key_obj = json.loads(key_str)
        dataset_path = key_obj["dataset_path"]
        dataset_name = key_obj["dataset_name"]
        dataset_kwargs = key_obj.get("dataset_kwargs") or {}

        # Only pass kwargs that the datasets version supports to avoid false negatives
        # from harness task configs using newer/older args.
        filtered_kwargs = {k: v for k, v in dataset_kwargs.items() if k in supported_kwargs}

        t0 = time.time()
        ok = False
        err: Exception | None = None
        try:
            _ = datasets.load_dataset_builder(
                dataset_path,
                dataset_name,
                **filtered_kwargs,
            )
            ok = True
        except Exception as e:  # noqa: BLE001
            err = e
            ok = False
            if args.fail_fast:
                raise
        dt_s = time.time() - t0

        res = {
            "idx": idx,
            "total": len(keys),
            "key": key_obj,
            "filtered_kwargs": filtered_kwargs,
            "ok": ok,
            "seconds": dt_s,
        }
        if not ok and err is not None:
            res["error_type"] = type(err).__name__
            res["error"] = str(err)
            failures.append(res)
        builder_results.append(res)

    # Summaries.
    total_tasks = len(task_to_key)
    total_unique_builders = len(keys)
    failing_builder_keys = {json.dumps(f["key"], sort_keys=True) for f in failures}

    failing_tasks: list[str] = []
    for key_str in failing_builder_keys:
        failing_tasks.extend(key_to_tasks.get(key_str, []))

    failure_reason_counts = Counter((f.get("error_type") or "Unknown") for f in failures)

    report = {
        "env": env,
        "summary": {
            "total_tasks_indexed": total_tasks,
            "yaml_parse_error_tasks": len(yaml_parse_errors),
            "tasks_missing_dataset_path": len(tasks_missing_dataset_path),
            "unique_dataset_builder_keys_probed": total_unique_builders,
            "builder_failures": len(failures),
            "tasks_impacted_by_builder_failures": len(set(failing_tasks)),
        },
        "failure_reason_counts": dict(failure_reason_counts),
        "yaml_parse_errors": yaml_parse_errors,
        "tasks_missing_dataset_path": sorted(tasks_missing_dataset_path),
        "builder_results": builder_results,
        "builder_failures": failures,
        "failing_tasks": sorted(set(failing_tasks)),
        "key_to_tasks": {k: sorted(v) for k, v in key_to_tasks.items()},
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    # Markdown summary: show biggest failure clusters.
    cluster_sizes: list[tuple[int, str]] = []
    for f in failures:
        k = json.dumps(f["key"], sort_keys=True)
        cluster_sizes.append((len(key_to_tasks.get(k, [])), k))
    cluster_sizes.sort(reverse=True)

    lines: list[str] = []
    lines.append("# lm-eval datasets==4.x task audit")
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- python: `{env['python'].splitlines()[0]}`")
    lines.append(f"- datasets: `{env['datasets_version']}`")
    lines.append(f"- lm_eval (pip): `{env['lm_eval_version']}`")
    lines.append("")
    lines.append("## Summary")
    for k, v in report["summary"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Failure reasons (dataset builder)")
    for reason, cnt in failure_reason_counts.most_common():
        lines.append(f"- **{reason}**: {cnt}")
    lines.append("")
    lines.append("## Largest failure clusters (top 20)")
    for size, k in cluster_sizes[:20]:
        key_obj = json.loads(k)
        lines.append(f"- **{size} tasks**: `{key_obj['dataset_path']}` / `{key_obj['dataset_name']}`")
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- This audit uses `datasets.load_dataset_builder()` only (no full dataset download)."
    )
    lines.append(
        "- If a task relies on custom `dataset_kwargs` and they were filtered out for "
        "compatibility, it may need a follow-up full-run validation."
    )
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines))

    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Wrote: {out_dir / 'report.json'}")
    print(f"Wrote: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()


