"""Authoring path for named benchmarks: validate, pin, optionally publish.

Definition model is intentionally small:

1. Freeze the full config and assets (including ``runtime.max_sessions``).
2. Optionally declare free variables (``knobs``) -- the only run-time exceptions.
3. Compute one workload fingerprint at the defaults over that session count.
4. Verify each free variable leaves that fingerprint unchanged.
5. Optionally publish the self-contained tree to the Hub.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import yaml
from vidhi.utils import create_class_from_dict

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.benchmark_define import BenchmarkDefineConfig
from veeksha.core.seeding import SeedManager
from veeksha.core.workload_fingerprint import WorkloadFingerprint
from veeksha.generator.session.registry import SessionGeneratorRegistry
from veeksha.logger import init_logger
from veeksha.named_benchmark.hub import (
    default_repo,
    definition_dir,
    load_definition,
    publish_benchmark,
)
from veeksha.named_benchmark.knobs import (
    KnobDeclarationError,
    KnobSpec,
    apply_knobs,
    parse_knob_specs,
    resolve_knob_values,
)
from veeksha.provenance import capture_environment, file_digest

logger = init_logger(__name__)


class BenchmarkDefineError(RuntimeError):
    """Definition failed validation or pin verification."""


def _require_keys(definition: dict[str, Any]) -> None:
    for key in ("name", "config"):
        if key not in definition:
            raise BenchmarkDefineError(
                f"benchmark definition missing required key '{key}'"
            )
    if not isinstance(definition["name"], str) or not definition["name"].strip():
        raise BenchmarkDefineError("'name' must be a non-empty string")
    if not isinstance(definition["config"], dict):
        raise BenchmarkDefineError("'config' must be a mapping (or !include one)")


def _build_benchmark_config(
    config_dict: dict[str, Any],
    *,
    seed: Optional[int] = None,
) -> BenchmarkConfig:
    """Materialize a BenchmarkConfig for generation-only use.

    Leaves ``runtime.max_sessions`` as declared in the definition — it is an
    ordinary frozen field, not overridden by the define CLI.
    """
    data = copy.deepcopy(config_dict)
    if seed is not None:
        data["seed"] = seed
    # No live target needed for generation-only; keep defaults.
    data.pop("server", None)
    return create_class_from_dict(BenchmarkConfig, data)


def _pin_session_count(benchmark_config: BenchmarkConfig) -> int:
    """Return how many sessions define must generate for the pin.

    Named benchmarks require a finite, positive ``runtime.max_sessions`` so the
    pin is the same stream a run will produce (not an ad-hoc sample size).
    """
    n = int(benchmark_config.runtime.max_sessions)
    if n <= 0:
        raise BenchmarkDefineError(
            "named benchmarks must set runtime.max_sessions to a positive "
            f"count (got {n}). It is a normal frozen config field — put it in "
            "the definition's config, not on the define CLI."
        )
    return n


def _generate_fingerprint(
    benchmark_config: BenchmarkConfig,
    *,
    max_sessions: int,
) -> WorkloadFingerprint:
    seed_manager = SeedManager(benchmark_config.seed)
    tokenizer_provider = benchmark_config.client.build_tokenizer_provider()
    session_generator = SessionGeneratorRegistry.get(
        benchmark_config.session_generator.get_type(),
        config=benchmark_config.session_generator,
        seed_manager=seed_manager,
        tokenizer_provider=tokenizer_provider,
    )
    fingerprint = WorkloadFingerprint()
    for _ in range(max_sessions):
        try:
            session = session_generator.generate_session()
        except StopIteration:
            break
        fingerprint.add_session(session)
    if fingerprint.session_count == 0:
        raise BenchmarkDefineError(
            "generation produced zero sessions; cannot pin an empty workload"
        )
    if fingerprint.session_count < max_sessions:
        logger.warning(
            "Generator exhausted after %d sessions (definition asks for %d); "
            "pinning the shorter stream",
            fingerprint.session_count,
            max_sessions,
        )
    return fingerprint


def _alternate_value(spec: KnobSpec, current: Any) -> Any:
    """Pick one alternate value to verify a free variable is pin-stable."""
    if spec.choices:
        for choice in spec.choices:
            if choice != current:
                return choice
        raise BenchmarkDefineError(
            f"knob '{spec.name}' has no alternate choice to verify against "
            f"(only {spec.choices!r}); free variables need at least two values "
            "so define can check they leave the workload unchanged"
        )
    if spec.type_name == "bool":
        return not bool(current)
    if spec.type_name == "int":
        return int(current) + 1
    if spec.type_name == "float":
        return float(current) * 2.0 if float(current) != 0 else 1.0
    if spec.type_name == "str":
        return f"{current}__alt"
    raise BenchmarkDefineError(
        f"knob '{spec.name}': cannot invent an alternate value for type "
        f"{spec.type_name}"
    )


def _verify_free_variables_are_pin_stable(
    base_config_dict: dict[str, Any],
    specs: tuple[KnobSpec, ...],
    base_values: dict[str, Any],
    base_fingerprint: str,
    *,
    max_sessions: int,
) -> None:
    """Each free variable must leave the workload fingerprint unchanged.

    If flipping a free variable moves the stream, it is not a free variable --
    freeze it in the definition or publish a separate definition for that setup.
    """
    for spec in specs:
        alt = _alternate_value(spec, base_values[spec.name])
        values = dict(base_values)
        values[spec.name] = alt
        cfg_dict = apply_knobs(base_config_dict, specs, values)
        cfg = _build_benchmark_config(cfg_dict)
        other = _generate_fingerprint(cfg, max_sessions=max_sessions).digest()
        if other != base_fingerprint:
            raise BenchmarkDefineError(
                f"free variable '{spec.name}' changes the workload fingerprint "
                f"when set to {alt!r} ({base_fingerprint} -> {other}). "
                "Only pin-stable flags may be free variables; freeze this in "
                "the definition or publish a separate definition for that setup."
            )


def _collect_assets(config_dict: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    session_gen = config_dict.get("session_generator") or {}
    if not isinstance(session_gen, dict):
        return assets
    trace_file = session_gen.get("trace_file")
    if isinstance(trace_file, str) and trace_file:
        path = Path(trace_file)
        if not path.is_absolute():
            path = (root / path).resolve()
        digest = file_digest(path)
        assets.append({"path": str(trace_file), "digest": digest})
    flavor = session_gen.get("flavor") or {}
    if isinstance(flavor, dict):
        local_path = flavor.get("local_path")
        if isinstance(local_path, str) and local_path and os.path.isfile(local_path):
            assets.append({"path": local_path, "digest": file_digest(local_path)})
    return assets


def define_benchmark(config: BenchmarkDefineConfig) -> dict[str, Any]:
    """Validate, pin, and optionally publish a named benchmark definition."""
    def_path = Path(config.definition)
    if def_path.is_dir():
        yml = def_path / "benchmark.yml"
    else:
        yml = def_path
        def_path = definition_dir(def_path)
    if not yml.is_file():
        raise BenchmarkDefineError(f"benchmark.yml not found at {yml}")

    definition = load_definition(yml)
    _require_keys(definition)
    name = definition["name"].strip()
    try:
        specs = parse_knob_specs(definition.get("knobs"))
    except KnobDeclarationError as exc:
        raise BenchmarkDefineError(str(exc)) from exc

    base_values = resolve_knob_values(specs)
    config_dict = apply_knobs(definition["config"], specs, base_values)
    benchmark_config = _build_benchmark_config(config_dict)
    max_sessions = _pin_session_count(benchmark_config)

    logger.info(
        "Computing workload fingerprint for %s (%d sessions from "
        "runtime.max_sessions)...",
        name,
        max_sessions,
    )
    fingerprint = _generate_fingerprint(benchmark_config, max_sessions=max_sessions)
    base_digest = fingerprint.digest()
    logger.info(
        "Workload fingerprint %s over %d sessions / %d requests",
        base_digest,
        fingerprint.session_count,
        fingerprint.request_count,
    )

    if specs:
        logger.info(
            "Verifying %d free variable(s) leave the fingerprint unchanged...",
            len(specs),
        )
        _verify_free_variables_are_pin_stable(
            definition["config"],
            specs,
            base_values,
            base_digest,
            max_sessions=max_sessions,
        )

    tokenizer_provider = benchmark_config.client.build_tokenizer_provider()
    environment = capture_environment()
    pins = {
        # Single pin: free variables must not change the workload, so there is
        # only one fingerprint for the definition.
        "workload_fingerprint": base_digest,
        "assets": _collect_assets(definition["config"], def_path),
        "tokenizer": {
            "model": getattr(tokenizer_provider, "model_name", None),
            "transformers": environment["packages"].get("transformers"),
            "tokenizers": environment["packages"].get("tokenizers"),
        },
        "veeksha": environment["veeksha"],
        "knob_defaults": base_values,
        "sessions_sampled": fingerprint.session_count,
        "fingerprint_version": fingerprint.summary()["fingerprint_version"],
    }
    definition["pins"] = pins
    definition.setdefault("version", 1)

    out_dir = Path(config.output) if config.output else def_path
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() != def_path.resolve():
        for item in def_path.iterdir():
            dest = out_dir / item.name
            if dest.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    out_yml = out_dir / "benchmark.yml"
    with open(out_yml, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            definition,
            handle,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    pins_path = out_dir / "pins.json"
    with open(pins_path, "w", encoding="utf-8") as handle:
        json.dump(pins, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logger.info("Wrote pins to %s and %s", out_yml, pins_path)

    if config.publish:
        repo = config.repo or default_repo()
        publish_benchmark(
            out_dir,
            name,
            repo=repo,
            private=config.private,
            commit_message=config.commit_message or None,
            tag=config.tag or None,
        )
        logger.info("Published %s to %s", name, repo)

    return {
        "name": name,
        "pins": pins,
        "output_dir": str(out_dir),
    }


def run_benchmark_define_cli(configs: list[BenchmarkDefineConfig]) -> None:
    """CLI entry point for ``veeksha benchmark define``."""
    for config in configs:
        result = define_benchmark(config)
        print(
            f"Pinned benchmark {result['name']}: "
            f"{result['pins']['workload_fingerprint']}"
        )
