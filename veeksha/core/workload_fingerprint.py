"""Content hash over the request stream a benchmark actually generated.

Two runs of the same config do not necessarily send the same requests. The
config names a trace *path*, not its contents; Hugging Face dataset loads can
follow a moving branch; and prompt text is built by encoding and decoding
through a tokenizer, so a ``transformers`` upgrade can rewrite it. A config hash
therefore proves nothing about the workload.

This module hashes the materialized workload instead: every generated session,
in generation order, reduced to a canonical form. Equal fingerprints mean the
same requests were produced; a mismatch is real drift, and the run manifest's
recorded inputs say which one moved.

Deliberately excluded from the hash:

``history``
    Populated at dispatch by the scheduler from completed responses
    (``traffic/concurrent.py:_populate_history``), not at generation. Including
    it would make the fingerprint depend on server replies.
``dispatch_ticket``
    Assigned by the dispatcher at runtime.

Ordering is caller-supplied and matters. Feed sessions in generation order --
``PrefetchWorker._generate_session`` already serializes generation behind
``generator_lock``, so hashing there is stable regardless of worker count.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Optional

from veeksha.core.request import Request
from veeksha.core.session import Session
from veeksha.core.session_graph import SessionGraph
from veeksha.provenance import file_digest

# Channel content fields that name an external asset rather than carrying it
# inline. These are hashed by file contents when they resolve to a real file,
# so a regenerated clip at the same path changes the fingerprint.
_ASSET_FIELDS = ("input_audio", "input_image", "input_video")

_FINGERPRINT_VERSION = 1


def serialize_channel_content(content: Any) -> dict[str, Any]:
    """Reduce channel content to a plain dict.

    Shared with ``TraceRecorder`` so the dispatch trace and the fingerprint
    cannot disagree about what a channel contains.
    """
    if is_dataclass(content) and not isinstance(content, type):
        return asdict(content)
    try:
        return dict(vars(content))
    except TypeError:
        return {"raw_str": str(content)}


def _canonical_asset(value: Any) -> Any:
    """Replace an asset reference with a digest of its contents when possible.

    Falls back to the reference itself for URLs, inline payloads, or files that
    cannot be read -- a fingerprint over the reference is still better than
    silently dropping the field, and the run manifest records the asset digests
    separately.
    """
    if not isinstance(value, str) or not value:
        return value
    try:
        if not os.path.isfile(value):
            return value
    except (OSError, ValueError):
        return value
    digest = file_digest(value)
    return digest if digest is not None else value


def _canonical_channels(request: Request) -> dict[str, Any]:
    channels: dict[str, Any] = {}
    for modality, content in request.channels.items():
        payload = serialize_channel_content(content)
        for asset_field in _ASSET_FIELDS:
            if asset_field in payload:
                payload[asset_field] = _canonical_asset(payload[asset_field])
        channels[str(getattr(modality, "name", modality)).lower()] = payload
    return channels


def _canonical_request(node_id: int, request: Request) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "request_id": request.id,
        "channels": _canonical_channels(request),
        "metadata": request.metadata,
        "requested_output": (
            serialize_channel_content(request.requested_output)
            if request.requested_output is not None
            else None
        ),
        "session_context": request.session_context,
    }


def _canonical_graph(graph: SessionGraph) -> dict[str, Any]:
    nodes = [
        {"id": node_id, "wait_after_ready": graph.nodes[node_id].wait_after_ready}
        for node_id in sorted(graph.nodes)
    ]
    # Read edges from `outgoing` only: `incoming` holds the same objects, so
    # including both would double-count without adding information.
    edges = sorted(
        (
            {
                "src": edge.src,
                "dst": edge.dst,
                "is_history_parent": edge.is_history_parent,
            }
            for edge_list in graph.outgoing.values()
            for edge in edge_list
        ),
        key=lambda item: (item["src"], item["dst"], item["is_history_parent"]),
    )
    return {"nodes": nodes, "edges": edges}


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(value)).hexdigest()
    return str(value)


def canonical_session_bytes(session: Session) -> bytes:
    """Return the canonical byte encoding hashed for one session."""
    payload = {
        "session_id": session.id,
        "graph": _canonical_graph(session.session_graph),
        "requests": [
            _canonical_request(node_id, session.requests[node_id])
            for node_id in sorted(session.requests)
        ],
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode("utf-8")


class WorkloadFingerprint:
    """Incremental hash over generated sessions, in generation order.

    Not thread-safe by design: callers feed it from inside the lock that already
    serializes session generation, and adding a second lock here would only
    invite holding two.
    """

    def __init__(self) -> None:
        self._hasher = hashlib.blake2b(digest_size=32)
        self._hasher.update(f"veeksha-workload-v{_FINGERPRINT_VERSION}\0".encode())
        self._session_count = 0
        self._request_count = 0

    def add_session(self, session: Session) -> None:
        encoded = canonical_session_bytes(session)
        # Length-prefix each session so concatenation is unambiguous: without
        # it, differently-split session boundaries could hash identically.
        self._hasher.update(len(encoded).to_bytes(8, "big"))
        self._hasher.update(encoded)
        self._session_count += 1
        self._request_count += len(session.requests)

    @property
    def session_count(self) -> int:
        return self._session_count

    @property
    def request_count(self) -> int:
        return self._request_count

    def digest(self) -> str:
        """Return the fingerprint as ``blake2b:<hex>``."""
        return f"blake2b:{self._hasher.hexdigest()}"

    def summary(self) -> dict[str, Any]:
        return {
            "workload_fingerprint": self.digest(),
            "fingerprint_version": _FINGERPRINT_VERSION,
            "sessions": self._session_count,
            "requests": self._request_count,
        }


def fingerprint_sessions(sessions: Any) -> str:
    """Convenience helper: fingerprint an iterable of sessions."""
    fingerprint = WorkloadFingerprint()
    for session in sessions:
        fingerprint.add_session(session)
    return fingerprint.digest()


def describe_drift(
    expected: Optional[dict[str, Any]], actual: dict[str, Any]
) -> list[str]:
    """Explain which recorded inputs differ between two run records.

    A fingerprint mismatch on its own is not actionable. Comparing the recorded
    inputs alongside it turns "two hashes differ" into "the trace contents
    changed" or "transformers was upgraded".
    """
    if not expected:
        return []

    reasons: list[str] = []

    def compare(label: str, path: tuple[str, ...]) -> None:
        old: Any = expected
        new: Any = actual
        for key in path:
            old = old.get(key) if isinstance(old, dict) else None
            new = new.get(key) if isinstance(new, dict) else None
        if old != new:
            reasons.append(f"{label}: {old!r} -> {new!r}")

    compare("veeksha git commit", ("veeksha", "git_commit"))
    compare("veeksha version", ("veeksha", "version"))
    for package in ("transformers", "tokenizers", "numpy", "datasets"):
        compare(f"{package} version", ("packages", package))
    compare("tokenizer model", ("tokenizer", "model"))
    compare("seed", ("seed",))

    # Keys are coerced to str: this runs while explaining a mismatch, so a
    # malformed record must not raise and mask the fingerprint error itself.
    def _asset_digests(record: dict[str, Any]) -> dict[str, Any]:
        return {
            str(asset.get("path")): asset.get("digest")
            for asset in (record.get("assets") or [])
            if isinstance(asset, dict)
        }

    expected_assets = _asset_digests(expected)
    actual_assets = _asset_digests(actual)
    for path in sorted(set(expected_assets) | set(actual_assets)):
        if expected_assets.get(path) != actual_assets.get(path):
            reasons.append(
                f"asset {path}: {expected_assets.get(path)!r} -> "
                f"{actual_assets.get(path)!r}"
            )

    return reasons
