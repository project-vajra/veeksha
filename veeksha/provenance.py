"""Environment provenance captured alongside every benchmark run.

A benchmark result is only interpretable if you know what produced it. Veeksha
generates its own workload -- prompt text is built by encoding and decoding
through a tokenizer to hit target token lengths -- so the veeksha revision and
the installed tokenizer stack are workload inputs, not incidental environment.
Recording them is what lets a later run explain *why* a workload fingerprint
moved instead of only reporting that it did.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any, Optional

from veeksha.version import __version__

# Packages whose version can change the generated workload or the recorded
# metrics. transformers/tokenizers drive prompt construction; numpy backs the
# seeded generators; ddsketch decides percentile estimation.
_TRACKED_DISTRIBUTIONS = (
    "transformers",
    "tokenizers",
    "numpy",
    "ddsketch",
    "datasets",
    "librosa",
    "soundfile",
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

_FILE_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


def git_commit(repo_root: Path = _REPO_ROOT) -> Optional[str]:
    """Return the HEAD commit of ``repo_root``, or None outside a checkout."""
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def git_is_dirty(repo_root: Path = _REPO_ROOT) -> Optional[bool]:
    """Return whether the checkout has uncommitted changes, or None if unknown.

    A dirty checkout means the recorded commit does not fully describe the code
    that ran, so a fingerprint mismatch against that commit is expected rather
    than alarming.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def distribution_version(name: str) -> Optional[str]:
    """Return an installed distribution's version, or None when absent."""
    try:
        return _dist_version(name)
    except PackageNotFoundError:
        return None


def file_digest(path: str | os.PathLike[str]) -> Optional[str]:
    """Return ``sha256:<hex>`` for a file's contents, or None if unreadable.

    Memoised on ``(path, size, mtime_ns)``: a benchmark can reference the same
    clip from thousands of requests, and re-reading each one would dominate
    workload generation. The size and mtime in the key mean a file rewritten in
    place is re-read rather than served stale from the cache.
    """
    try:
        resolved = Path(path)
        stat = resolved.stat()
    except OSError:
        return None

    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    cached = _FILE_DIGEST_CACHE.get(key)
    if cached is not None:
        return cached

    hasher = hashlib.sha256()
    try:
        with open(resolved, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError:
        return None

    digest = f"sha256:{hasher.hexdigest()}"
    _FILE_DIGEST_CACHE[key] = digest
    return digest


def python_environment() -> dict[str, Any]:
    """Describe the interpreter, including whether the GIL is disabled.

    Veeksha targets free-threaded Python, and GIL state changes the concurrency
    behaviour a run actually exercised, so it belongs in the record.
    """
    gil_enabled = getattr(sys, "_is_gil_enabled", None)
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "free_threaded": bool(getattr(sys, "abiflags", "").find("t") >= 0)
        or sys.version.find("free-threading") >= 0,
        "gil_enabled": gil_enabled() if callable(gil_enabled) else None,
    }


def capture_environment() -> dict[str, Any]:
    """Capture the provenance recorded at the start of every run."""
    return {
        "veeksha": {
            "version": __version__,
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
        },
        "python": python_environment(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(),
        },
        "packages": {
            name: distribution_version(name) for name in _TRACKED_DISTRIBUTIONS
        },
    }
