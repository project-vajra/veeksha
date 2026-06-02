"""Typed configuration model for sweep planning."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional, Tuple

TRACE_SHAREGPT = "sharegpt"
TRACE_SEED_TTS = "seed_tts"
TRACE_ALIASES = {
    "sharegpt": TRACE_SHAREGPT,
    "share_gpt": TRACE_SHAREGPT,
    "seed_tts": TRACE_SEED_TTS,
    "seedtts": TRACE_SEED_TTS,
    "seed_tts_text": TRACE_SEED_TTS,
}

DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_COOLDOWN_SECONDS = 120
DEFAULT_MAX_SESSIONS = 20000
DEFAULT_MIN_TOKENS = 20
DEFAULT_MAX_TOKENS = 150


class SweepConfigError(ValueError):
    """Raised when a programmatic sweep config is invalid."""


@dataclass(frozen=True)
class SweepConfig:
    """Typed sweep configuration shared by CLI and programmatic callers."""

    sweep_type: str
    engine: str
    model: str
    trace: str = TRACE_SHAREGPT
    output_dir: Optional[str] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    min_chars: Optional[int] = None
    max_chars: Optional[int] = None
    concurrencies: Optional[Tuple[int, ...]] = None
    concurrency: Optional[int] = None
    sizes: Optional[Tuple[int, ...]] = None
    range_start: Optional[int] = None
    range_end: Optional[int] = None
    step: Optional[int] = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS
    max_sessions: int = DEFAULT_MAX_SESSIONS

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SweepConfig":
        unknown = sorted(set(raw) - {field.name for field in fields(cls)})
        if unknown:
            raise SweepConfigError(
                f"unsupported sweep config keys: {', '.join(unknown)}"
            )

        trace = _optional_str(raw, "trace")
        if trace is None:
            trace = TRACE_SHAREGPT

        timeout_seconds = _optional_int(raw, "timeout_seconds")
        if timeout_seconds is None:
            timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        cooldown_seconds = _optional_int(raw, "cooldown_seconds")
        if cooldown_seconds is None:
            cooldown_seconds = DEFAULT_COOLDOWN_SECONDS
        max_sessions = _optional_int(raw, "max_sessions")
        if max_sessions is None:
            max_sessions = DEFAULT_MAX_SESSIONS

        return cls(
            sweep_type=_required_str(raw, "sweep_type"),
            engine=_required_str(raw, "engine"),
            model=_required_str(raw, "model"),
            trace=_normalize_trace(trace),
            output_dir=_optional_str(raw, "output_dir"),
            min_tokens=_optional_int(raw, "min_tokens"),
            max_tokens=_optional_int(raw, "max_tokens"),
            min_chars=_optional_int(raw, "min_chars"),
            max_chars=_optional_int(raw, "max_chars"),
            concurrencies=_optional_int_tuple(raw, "concurrencies"),
            concurrency=_optional_int(raw, "concurrency"),
            sizes=_optional_int_tuple(raw, "sizes"),
            range_start=_optional_int(raw, "range_start"),
            range_end=_optional_int(raw, "range_end"),
            step=_optional_int(raw, "step"),
            timeout_seconds=timeout_seconds,
            cooldown_seconds=cooldown_seconds,
            max_sessions=max_sessions,
        )


def _normalize_trace(raw: str) -> str:
    key = raw.strip().lower().replace("-", "_")
    if key not in TRACE_ALIASES:
        supported = ", ".join(sorted(TRACE_ALIASES))
        raise SweepConfigError(f"trace must be one of: {supported}")
    return TRACE_ALIASES[key]


def _required_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SweepConfigError(f"sweep.{key} must be a non-empty string")
    return value


def _optional_str(raw: Mapping[str, Any], key: str) -> Optional[str]:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SweepConfigError(f"sweep.{key} must be a string")
    return value


def _optional_int(raw: Mapping[str, Any], key: str) -> Optional[int]:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SweepConfigError(f"sweep.{key} must be an integer")
    return value


def _optional_int_tuple(raw: Mapping[str, Any], key: str) -> Optional[Tuple[int, ...]]:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise SweepConfigError(f"sweep.{key} must be a list of integers")
    return tuple(value)
