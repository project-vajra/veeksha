"""Free variables a benchmark definition allows at run time.

A named benchmark freezes its full config and assets at definition time. The
only things an evaluator may change are the free variables declared here
(``knobs`` in ``benchmark.yml``) -- concurrency, target QPS, and so on.

Everything else is pinned. Free variables are also required to leave the
workload fingerprint unchanged: ``veeksha benchmark define`` flips each one to
an alternate value and fails if the generated request stream moves. If a flag
changes the stream, it is not a free variable -- freeze it in the definition
or publish a separate definition.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from vidhi import field as vidhi_field
from vidhi import frozen_dataclass
from vidhi.nested_config import _deep_merge, _flat_to_nested

# Scalar types that survive a YAML round-trip and map cleanly onto a CLI flag.
# Richer types belong in the pinned config, not the free-variable surface.
_TYPES: dict[str, type] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}

_REQUIRED_KEYS = ("target", "type", "default")
_ALLOWED_KEYS = (
    "target",
    "type",
    "default",
    "help",
    "choices",
)


class KnobDeclarationError(ValueError):
    """A benchmark definition declares its free variables incorrectly."""


@dataclass(frozen=True)
class KnobSpec:
    """One declared free variable (run-time exception to the frozen config)."""

    name: str
    target: str
    type_name: str
    default: Any
    help: str
    choices: Optional[tuple[Any, ...]]

    @property
    def py_type(self) -> type:
        return _TYPES[self.type_name]

    def coerce(self, value: Any) -> Any:
        """Coerce a value to the declared type and check it against choices."""
        if self.type_name == "bool" and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                coerced: Any = True
            elif lowered in ("false", "0", "no", "off"):
                coerced = False
            else:
                raise KnobDeclarationError(
                    f"knob '{self.name}': cannot read {value!r} as a boolean"
                )
        else:
            try:
                coerced = self.py_type(value)
            except (TypeError, ValueError) as error:
                raise KnobDeclarationError(
                    f"knob '{self.name}': cannot read {value!r} as "
                    f"{self.type_name} ({error})"
                ) from error

        if self.choices is not None and coerced not in self.choices:
            allowed = ", ".join(repr(choice) for choice in self.choices)
            raise KnobDeclarationError(
                f"knob '{self.name}': {coerced!r} is not one of [{allowed}]"
            )
        return coerced


def parse_knob_specs(raw: Optional[Mapping[str, Any]]) -> tuple[KnobSpec, ...]:
    """Validate the ``knobs`` block of a benchmark definition."""
    if not raw:
        return ()
    if not isinstance(raw, Mapping):
        raise KnobDeclarationError(
            f"'knobs' must be a mapping of name -> declaration, got "
            f"{type(raw).__name__}"
        )

    specs: list[KnobSpec] = []
    for name, declaration in raw.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise KnobDeclarationError(
                f"knob name {name!r} must be a valid Python identifier: it "
                "becomes both a config field and a CLI flag"
            )
        if not isinstance(declaration, Mapping):
            raise KnobDeclarationError(
                f"knob '{name}': declaration must be a mapping, got "
                f"{type(declaration).__name__}"
            )

        unknown = sorted(set(declaration) - set(_ALLOWED_KEYS))
        if unknown:
            raise KnobDeclarationError(
                f"knob '{name}': unknown keys {unknown}. "
                f"Allowed: {sorted(_ALLOWED_KEYS)}"
            )
        missing = [key for key in _REQUIRED_KEYS if key not in declaration]
        if missing:
            raise KnobDeclarationError(f"knob '{name}': missing keys {missing}")

        type_name = declaration["type"]
        if type_name not in _TYPES:
            raise KnobDeclarationError(
                f"knob '{name}': unsupported type {type_name!r}. "
                f"Supported: {sorted(_TYPES)}"
            )

        target = declaration["target"]
        if not isinstance(target, str) or not target.strip():
            raise KnobDeclarationError(
                f"knob '{name}': 'target' must be a dotted config path"
            )
        if target.startswith(".") or target.endswith(".") or ".." in target:
            raise KnobDeclarationError(
                f"knob '{name}': malformed target path {target!r}"
            )

        raw_choices = declaration.get("choices")
        if raw_choices is not None and not isinstance(raw_choices, (list, tuple)):
            raise KnobDeclarationError(
                f"knob '{name}': 'choices' must be a list when provided"
            )

        # Coerce choices and default so a definition cannot declare
        # `type: int` with `choices: ["8"]` and fail only at run time.
        probe = KnobSpec(
            name=name,
            target=target.strip(),
            type_name=type_name,
            default=declaration["default"],
            help=str(declaration.get("help") or ""),
            choices=None,
        )
        choices = (
            tuple(probe.coerce(choice) for choice in raw_choices)
            if raw_choices is not None
            else None
        )
        specs.append(
            KnobSpec(
                name=name,
                target=probe.target,
                type_name=type_name,
                default=probe.coerce(probe.default),
                help=probe.help,
                choices=choices,
            )
        )

    targets: dict[str, str] = {}
    for spec in specs:
        clash = targets.get(spec.target)
        if clash is not None:
            raise KnobDeclarationError(
                f"knobs '{clash}' and '{spec.name}' both target "
                f"'{spec.target}'; the later would silently win"
            )
        targets[spec.target] = spec.name

    return tuple(specs)


def build_knobs_config_class(specs: Sequence[KnobSpec]) -> type:
    """Materialize declared free variables into a real vidhi config class."""
    namespace: dict[str, Any] = {
        "__module__": __name__,
        "__qualname__": "BenchmarkKnobs",
        "__doc__": "Free variables declared by a benchmark definition.",
        "__annotations__": {},
    }
    for spec in specs:
        namespace["__annotations__"][spec.name] = spec.py_type
        namespace[spec.name] = vidhi_field(
            spec.default,
            help=spec.help or f"Benchmark free variable '{spec.name}' -> {spec.target}",
            choices=list(spec.choices) if spec.choices is not None else None,
        )
    return frozen_dataclass(type("BenchmarkKnobs", (), namespace))


def add_knob_arguments(
    parser: argparse.ArgumentParser, specs: Sequence[KnobSpec]
) -> None:
    """Expose each free variable as a CLI flag."""
    for spec in specs:
        help_text = spec.help or f"-> {spec.target}"
        parser.add_argument(
            f"--{spec.name}",
            dest=spec.name,
            type=str,
            default=None,
            choices=(
                [str(choice) for choice in spec.choices]
                if spec.choices is not None
                else None
            ),
            help=f"{help_text} (default: {spec.default})",
        )


def find_cli_option(args: Sequence[str], option: str) -> Optional[str]:
    """Return the value of ``--option`` / ``--option=value`` in ``args``, if any."""
    flag = f"--{option}"
    eq_prefix = f"{flag}="
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == flag:
            if i + 1 >= len(args) or args[i + 1].startswith("-"):
                raise KnobDeclarationError(f"{flag} requires a value")
            return args[i + 1]
        if arg.startswith(eq_prefix):
            return arg[len(eq_prefix) :]
        i += 1
    return None


def peel_knob_cli_args(
    args: Sequence[str], specs: Sequence[KnobSpec]
) -> tuple[list[str], dict[str, Any]]:
    """Strip free-variable flags from ``args`` and return ``(remaining, values)``.

    Supports ``--name value`` and ``--name=value``. Unknown flags are left in
    ``remaining`` for the main config parser. Values are coerced and choice-
    checked against the declaration.
    """
    by_name = {spec.name: spec for spec in specs}
    if not by_name:
        return list(args), {}

    remaining: list[str] = []
    values: dict[str, Any] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            body = arg[2:]
            if "=" in body:
                name, raw = body.split("=", 1)
                if name in by_name:
                    values[name] = by_name[name].coerce(raw)
                    i += 1
                    continue
            elif body in by_name:
                if i + 1 >= len(args) or str(args[i + 1]).startswith("-"):
                    raise KnobDeclarationError(f"--{body} requires a value")
                values[body] = by_name[body].coerce(args[i + 1])
                i += 2
                continue
        remaining.append(arg)
        i += 1
    return remaining, values


def resolve_knob_values(
    specs: Sequence[KnobSpec], provided: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """Return every free variable's effective value: default plus overrides."""
    provided = provided or {}
    known = {spec.name for spec in specs}
    unknown = sorted(set(provided) - known)
    if unknown:
        raise KnobDeclarationError(
            f"unknown free variable(s) {unknown}. This benchmark declares: "
            f"{sorted(known) or ['(none)']}"
        )

    values: dict[str, Any] = {}
    for spec in specs:
        raw = provided.get(spec.name)
        values[spec.name] = spec.default if raw is None else spec.coerce(raw)
    return values


def knob_overrides(
    specs: Sequence[KnobSpec], values: Mapping[str, Any]
) -> dict[str, Any]:
    """Turn resolved free-variable values into a nested config override dict."""
    flat = {spec.target: values[spec.name] for spec in specs if spec.name in values}
    return _flat_to_nested(flat) if flat else {}


def apply_knobs(
    config_dict: Mapping[str, Any],
    specs: Sequence[KnobSpec],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge free-variable values into a benchmark config dict.

    Free variables take precedence over the definition's pinned config: the
    pin supplies the frozen base, the free variables are the declared exceptions.
    """
    overrides = knob_overrides(specs, values)
    if not overrides:
        return dict(config_dict)
    return _deep_merge(dict(config_dict), overrides)
