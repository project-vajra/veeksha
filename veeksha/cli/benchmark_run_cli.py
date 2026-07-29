"""Parse ``veeksha benchmark run`` / ad-hoc run args, including free variables.

Free variables are declared per definition, so they cannot live on
``BenchmarkConfig`` as static fields. When ``--benchmark`` is set we load the
definition's knobs, peel ``--<knob>`` flags from the argv, parse the rest as a
normal ``BenchmarkConfig``, then stash the knob overrides for resolve.
"""

from __future__ import annotations

from typing import Any, Optional

from veeksha.benchmark_hub import load_definition
from veeksha.benchmark_knobs import (
    KnobDeclarationError,
    find_cli_option,
    parse_knob_specs,
    peel_knob_cli_args,
)
from veeksha.benchmark_resolve import _definition_path
from veeksha.cli.parsing import parse_cli_sweep
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


def _load_knob_specs(benchmark: str, revision: Optional[str]) -> tuple[Any, ...]:
    try:
        def_dir = _definition_path(benchmark, revision)
        definition = load_definition(def_dir / "benchmark.yml")
        return parse_knob_specs(definition.get("knobs"))
    except KnobDeclarationError as exc:
        raise SystemExit(f"Invalid free variables in {benchmark!r}: {exc}") from exc
    except Exception as exc:
        raise SystemExit(
            f"Could not load benchmark definition {benchmark!r}"
            + (f" (revision {revision})" if revision else "")
            + f": {exc}"
        ) from exc


def parse_benchmark_run_configs(args: list[str]) -> list[BenchmarkConfig]:
    """Parse run CLI args into one or more ``BenchmarkConfig`` instances.

    When ``--benchmark`` is present, free-variable flags declared by that
    definition (e.g. ``--concurrency 64``) are accepted, coerced, and attached
    as ``_knob_overrides`` for :func:`resolve_named_benchmark`.
    """
    try:
        benchmark = find_cli_option(args, "benchmark")
        revision = find_cli_option(args, "benchmark_revision")
    except KnobDeclarationError as exc:
        raise SystemExit(str(exc)) from exc

    remaining = list(args)
    knob_overrides: dict[str, Any] = {}
    if benchmark:
        specs = _load_knob_specs(benchmark, revision)
        try:
            remaining, knob_overrides = peel_knob_cli_args(remaining, specs)
        except KnobDeclarationError as exc:
            raise SystemExit(str(exc)) from exc
        if specs:
            logger.debug(
                "Accepted free-variable CLI overrides for %s: %s",
                benchmark,
                knob_overrides or "(defaults)",
            )

    configs = parse_cli_sweep(BenchmarkConfig, args=remaining)
    for cfg in configs:
        if knob_overrides:
            object.__setattr__(cfg, "_knob_overrides", dict(knob_overrides))
        # Mark free-variable names as CLI-provided so frozen-override checks
        # treat them as allowed.
        if knob_overrides:
            provided = set(getattr(cfg, "_cli_provided_keys", frozenset()) or ())
            provided.update(knob_overrides.keys())
            object.__setattr__(cfg, "_cli_provided_keys", frozenset(provided))
    return configs
