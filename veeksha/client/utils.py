"""Shared client-side helpers for paced streaming-text TTS.

Text segmentation (emulating an upstream LLM's per-token output), delta pacing
(emulating its decode cadence), and WebSocket error flattening/mapping are
transport-agnostic; they live here rather than in the WebSocket clients so each
client stays focused on its wire protocol and metric contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from websockets.exceptions import (
    ConnectionClosedError,
    InvalidHandshake,
    InvalidStatus,
)

from veeksha.config.generator.interval import (
    FixedIntervalGeneratorConfig,
    PoissonIntervalGeneratorConfig,
)
from veeksha.generator.interval.fixed import FixedIntervalGenerator
from veeksha.generator.interval.poisson import PoissonIntervalGenerator

if TYPE_CHECKING:
    from veeksha.config.client import TextPacingConfig


# ---------------------------------------------------------------------------
# Text segmentation (LLM-token emulation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextSegment:
    """Text for one paced Realtime conversation item.

    ``text`` is the exact substring streamed for this delta (whitespace
    preserved); ``n_tokens`` is the number of whitespace tokens it groups.
    """

    text: str
    n_chars: int
    n_tokens: int


def segment_text(text: str, tokens_per_delta: int) -> list[TextSegment]:
    """Split ``text`` into paced deltas of ``tokens_per_delta`` whitespace tokens.

    Each token is a ``\\S+`` run plus its trailing whitespace (via
    ``re.findall(r"\\S+\\s*")``); any leading whitespace is prepended to the
    first piece. Consecutive pieces are grouped ``tokens_per_delta`` at a time.

    HARD INVARIANT: ``"".join(s.text for s in segments) == text`` — the WER
    verification path reconstructs the exact input by concatenating deltas, so
    no character may be dropped or added.
    """
    if not text:
        return []

    pieces = re.findall(r"\S+\s*", text)
    if not pieces:
        # Whitespace-only text: emit it as a single (zero-token) segment so the
        # round-trip invariant still holds.
        return [TextSegment(text=text, n_chars=len(text), n_tokens=0)]

    # Prepend any leading whitespace (which ``\S+\s*`` does not capture) to the
    # first piece so the concatenation is byte-for-byte identical to ``text``.
    leading = text[: len(text) - len(text.lstrip())]
    if leading:
        pieces[0] = leading + pieces[0]

    segments: list[TextSegment] = []
    for start in range(0, len(pieces), tokens_per_delta):
        group = pieces[start : start + tokens_per_delta]
        chunk = "".join(group)
        segments.append(
            TextSegment(text=chunk, n_chars=len(chunk), n_tokens=len(group))
        )
    return segments


# ---------------------------------------------------------------------------
# Delta pacing (upstream decode-rate emulation)
# ---------------------------------------------------------------------------


class TextDeltaPacer:
    """Emits inter-delta gaps emulating an upstream LLM's decode cadence.

    ``mean_gap_s = tokens_per_delta / tokens_per_second``. A ``poisson``
    distribution jitters the gaps around that mean (seeded, so runs are
    reproducible); ``fixed`` returns the mean gap every time.
    """

    def __init__(self, pacing: "TextPacingConfig", seed: int) -> None:
        self.initial_delay_s = pacing.initial_delay_s
        mean_gap_s = pacing.tokens_per_delta / pacing.tokens_per_second
        if pacing.gap_distribution == "poisson":
            self._generator = PoissonIntervalGenerator(
                PoissonIntervalGeneratorConfig(arrival_rate=1.0 / mean_gap_s),
                np.random.RandomState(seed),
            )
        else:
            # FixedIntervalGenerator's signature declares ``rng: None``.
            self._generator = FixedIntervalGenerator(
                FixedIntervalGeneratorConfig(interval=mean_gap_s), None
            )

    def next_gap(self) -> float:
        """Return the next inter-delta gap in seconds."""
        return self._generator.get_next_interval()


# ---------------------------------------------------------------------------
# WebSocket error flattening / mapping
# ---------------------------------------------------------------------------

# Transport exceptions ranked most- to least-specific. Clients prepend their
# own server-error types so protocol errors win over the transport failures
# they cause (e.g. a server error that also closes the connection).
WS_TRANSPORT_ERROR_PRIORITY: tuple[type[BaseException], ...] = (
    InvalidStatus,
    InvalidHandshake,
    ConnectionClosedError,
    TimeoutError,
    OSError,
)


def flatten_ws_exception(
    exc: BaseException, priority: tuple[type[BaseException], ...]
) -> BaseException:
    """Reduce a (possibly nested) ExceptionGroup to its most specific leaf.

    ``asyncio.TaskGroup`` raises an ``ExceptionGroup`` bundling the failing
    task exceptions; connect-phase failures propagate bare. Collect the leaves
    and pick by ``priority`` so error mapping sees the exception that actually
    determines the error code.
    """
    if not isinstance(exc, BaseExceptionGroup):
        return exc

    leaves: list[BaseException] = []

    def _collect(node: BaseException) -> None:
        if isinstance(node, BaseExceptionGroup):
            for sub in node.exceptions:
                _collect(sub)
        else:
            leaves.append(node)

    _collect(exc)
    if not leaves:
        return exc

    for exc_type in priority:
        for leaf in leaves:
            if isinstance(leaf, exc_type):
                return leaf
    return leaves[0]


def map_ws_transport_error(exc: BaseException, timeout_msg: str) -> tuple[int, str]:
    """Map a flattened transport exception to an (error_code, error_msg) pair.

    Server-signalled protocol errors are client-specific and must be handled
    by the caller before falling through to this transport mapping.
    """
    if isinstance(exc, InvalidStatus):
        # Handshake rejected: surface the server's HTTP status code.
        return exc.response.status_code, str(exc)
    if isinstance(exc, TimeoutError):
        return 408, timeout_msg
    if isinstance(exc, (InvalidHandshake, OSError)):
        # Connect-phase failure (DNS/refused/handshake): unreachable server.
        return 503, str(exc)
    # ConnectionClosedError and anything else: unclassified transport failure.
    return 520, str(exc)
