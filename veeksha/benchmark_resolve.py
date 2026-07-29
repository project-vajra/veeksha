"""Resolve a named benchmark definition into a runnable BenchmarkConfig."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

from vidhi.utils import create_class_from_dict

from veeksha.benchmark_hub import fetch_benchmark, load_definition
from veeksha.benchmark_knobs import (
    KnobDeclarationError,
    KnobSpec,
    apply_knobs,
    parse_knob_specs,
    resolve_knob_values,
)
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


class NamedBenchmarkError(RuntimeError):
    """Failed to resolve or pin-check a named benchmark."""


# CLI keys always allowed when running a named (frozen) benchmark.
# Everything else is frozen unless it is a declared free-variable target/name
# or the user passes --allow_config_override.
_NAMED_RUN_ALLOWED_KEYS = frozenset(
    {
        "benchmark",
        "benchmark_revision",
        "allow_config_override",
        "allow_workload_drift",
        "output_dir",
    }
)
_NAMED_RUN_ALLOWED_PREFIXES = (
    "endpoint.",
    "server.",
    "wandb.",
)


def reject_frozen_overrides(
    provided_keys: Optional[frozenset[str] | set[str]],
    specs: tuple[KnobSpec, ...],
    *,
    allow_config_override: bool,
) -> None:
    """Error if the user overrode a field frozen by the named definition.

    Allowed without ``allow_config_override``:

    * identity / escape hatches (``benchmark``, ``allow_*``)
    * run placement (``output_dir``, ``endpoint.*``, ``server.*``, ``wandb.*``)
    * declared free variables (knob names and their ``target`` paths)
    """
    if allow_config_override or not provided_keys:
        return

    free_names = {spec.name for spec in specs}
    free_targets = {spec.target for spec in specs}

    forbidden: list[str] = []
    for key in sorted(provided_keys):
        if key in _NAMED_RUN_ALLOWED_KEYS:
            continue
        if any(key == p or key.startswith(p) for p in _NAMED_RUN_ALLOWED_PREFIXES):
            continue
        if key in free_names:
            continue
        if key in free_targets or any(
            key == t or key.startswith(f"{t}.") for t in free_targets
        ):
            continue
        forbidden.append(key)

    if not forbidden:
        return

    free_list = ", ".join(sorted(free_names)) or "(none)"
    raise NamedBenchmarkError(
        "Named benchmarks are frozen: these CLI flags override definition "
        f"fields and are not declared free variables: {', '.join(forbidden)}. "
        f"Declared free variables: {free_list}. "
        "Remove the flags, declare them as knobs in the definition, or pass "
        "--allow_config_override true (marks the run unpinned)."
    )


def _definition_path(benchmark: str, revision: Optional[str]) -> Path:
    """Return a local directory containing benchmark.yml for ``benchmark``."""
    candidate = Path(benchmark)
    if candidate.is_dir() and (candidate / "benchmark.yml").is_file():
        return candidate
    if candidate.is_file() and candidate.name == "benchmark.yml":
        return candidate.parent
    cache_root = Path(
        os.environ.get(
            "VEEKSHA_BENCHMARK_CACHE",
            Path.home() / ".cache" / "veeksha" / "benchmarks",
        )
    )
    return fetch_benchmark(
        benchmark,
        revision=revision,
        local_dir=cache_root,
        force=False,
    )


def resolve_named_benchmark(
    cfg: BenchmarkConfig,
    *,
    knob_overrides: Optional[dict[str, Any]] = None,
) -> tuple[BenchmarkConfig, dict[str, Any]]:
    """Fetch a named definition, apply free variables, return (config, meta).

    ``meta`` carries pins and resolved free-variable values for the run
    manifest and post-run fingerprint check.
    """
    if not cfg.benchmark:
        raise NamedBenchmarkError("resolve_named_benchmark requires cfg.benchmark")

    def_dir = _definition_path(cfg.benchmark, cfg.benchmark_revision)
    definition = load_definition(def_dir / "benchmark.yml")
    name = definition.get("name") or cfg.benchmark
    try:
        specs = parse_knob_specs(definition.get("knobs"))
    except KnobDeclarationError as exc:
        raise NamedBenchmarkError(str(exc)) from exc

    provided_keys = getattr(cfg, "_cli_provided_keys", None)
    reject_frozen_overrides(
        provided_keys,
        specs,
        allow_config_override=bool(cfg.allow_config_override),
    )

    provided = dict(knob_overrides or {})
    # Fall back to values stashed on the config during CLI peel.
    if not provided:
        stashed = getattr(cfg, "_knob_overrides", None)
        if isinstance(stashed, dict):
            provided = dict(stashed)
    values = resolve_knob_values(specs, provided)

    base_config = definition.get("config")
    if not isinstance(base_config, dict):
        raise NamedBenchmarkError(f"definition {name!r} has no config mapping")

    merged = apply_knobs(copy.deepcopy(base_config), specs, values)

    # Preserve CLI target/output fields that are not free variables.
    if cfg.server is not None:
        from veeksha.config.utils import to_serializable_config_dict

        merged["server"] = to_serializable_config_dict(cfg.server)
        merged.pop("endpoint", None)
    elif cfg.endpoint is not None:
        from veeksha.config.utils import to_serializable_config_dict

        merged["endpoint"] = to_serializable_config_dict(cfg.endpoint)
        merged.pop("server", None)

    merged["output_dir"] = cfg.output_dir
    merged["allow_config_override"] = cfg.allow_config_override
    merged["allow_workload_drift"] = cfg.allow_workload_drift
    merged["benchmark"] = name
    merged["benchmark_revision"] = cfg.benchmark_revision
    if cfg.wandb is not None:
        from veeksha.config.utils import to_serializable_config_dict

        merged["wandb"] = to_serializable_config_dict(cfg.wandb)

    resolved = create_class_from_dict(BenchmarkConfig, merged)

    pins = definition.get("pins") or {}
    meta = {
        "name": name,
        "version": definition.get("version"),
        "revision": cfg.benchmark_revision,
        "definition_dir": str(def_dir),
        "knobs": values,
        "pins": pins,
        "unpinned": bool(cfg.allow_config_override),
    }
    logger.info(
        "Resolved named benchmark %s (revision=%s) with free variables %s",
        name,
        cfg.benchmark_revision or "local/main",
        values,
    )
    return resolved, meta


def expected_fingerprint(meta: dict[str, Any]) -> Optional[str]:
    """Return the single pinned workload fingerprint, if present."""
    pins = meta.get("pins") or {}
    fp = pins.get("workload_fingerprint")
    if isinstance(fp, str) and fp:
        return fp
    return None


def check_workload_pin(
    *,
    actual_digest: str,
    named_meta: dict[str, Any],
    allow_workload_drift: bool,
    actual_inputs: Optional[dict[str, Any]] = None,
    stage: str = "finalize",
) -> None:
    """Compare the computed workload fingerprint to the definition pin.

    Raises :class:`NamedBenchmarkError` on mismatch unless
    ``allow_workload_drift`` is set (then logs a warning).
    """
    from veeksha.core.workload_fingerprint import describe_drift
    from veeksha.logger import init_logger

    log = init_logger(__name__)
    pinned = expected_fingerprint(named_meta)
    if not pinned or pinned == actual_digest:
        return

    expected_inputs = (named_meta.get("pins") or {}).get("inputs") or (
        named_meta.get("pins") or {}
    )
    reasons = describe_drift(expected_inputs, actual_inputs or {})
    detail = "; ".join(reasons) if reasons else "no input diffs recorded"
    message = (
        f"Workload fingerprint mismatch for named benchmark "
        f"{named_meta.get('name')!r} at {stage}: expected {pinned}, "
        f"got {actual_digest}. Likely causes: {detail}"
    )
    if allow_workload_drift:
        log.warning(message)
        return
    raise NamedBenchmarkError(message)
