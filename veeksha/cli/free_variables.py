"""Handle a named benchmark's free variables on the command line.

Free variables are declared per definition, so they cannot exist as static
fields on ``BenchmarkConfig`` and vidhi's parser has never heard of them. They
are peeled out of argv before parsing and re-attached to the parsed config
afterwards; everything in between is ordinary vidhi.
"""

from __future__ import annotations

from typing import Any, Optional

from veeksha.cli.parsing import parse_cli_sweep
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.logger import init_logger
from veeksha.named_benchmark.hub import load_definition
from veeksha.named_benchmark.knobs import (
    KnobDeclarationError,
    find_cli_option,
    parse_knob_specs,
    peel_knob_cli_args,
)
from veeksha.named_benchmark.resolve import _definition_path

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


def peel(args: list[str]) -> tuple[list[str], dict[str, Any]]:
    """Split declared ``--<knob>`` flags out of ``args``.

    A no-op unless ``--benchmark`` names a definition, so this is safe to call
    on any command's argv.
    """
    try:
        benchmark = find_cli_option(args, "benchmark")
        revision = find_cli_option(args, "benchmark_revision")
    except KnobDeclarationError as exc:
        raise SystemExit(str(exc)) from exc

    if not benchmark:
        return list(args), {}

    specs = _load_knob_specs(benchmark, revision)
    try:
        remaining, overrides = peel_knob_cli_args(list(args), specs)
    except KnobDeclarationError as exc:
        raise SystemExit(str(exc)) from exc

    if specs:
        logger.debug(
            "Accepted free-variable CLI overrides for %s: %s",
            benchmark,
            overrides or "(defaults)",
        )
    return remaining, overrides


def attach(configs: list[Any], overrides: dict[str, Any]) -> None:
    """Re-attach peeled free variables to the parsed benchmark configs.

    Their names are also recorded as CLI-provided so the frozen-override check
    treats them as allowed rather than as edits to a pinned field.
    """
    if not overrides:
        return
    for cfg in configs:
        if not isinstance(cfg, BenchmarkConfig):
            continue
        object.__setattr__(cfg, "_knob_overrides", dict(overrides))
        provided = set(getattr(cfg, "_cli_provided_keys", frozenset()) or ())
        provided.update(overrides.keys())
        object.__setattr__(cfg, "_cli_provided_keys", frozenset(provided))


def parse_benchmark_run_configs(args: list[str]) -> list[BenchmarkConfig]:
    """Parse args straight into ``BenchmarkConfig``s, free variables included.

    Used by the ``veeksha.benchmark`` module entry point, which bypasses the
    top-level command group.
    """
    remaining, overrides = peel(args)
    configs = parse_cli_sweep(BenchmarkConfig, args=remaining)
    attach(configs, overrides)
    return configs
