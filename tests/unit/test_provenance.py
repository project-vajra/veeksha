"""Unit tests for environment provenance capture."""

from __future__ import annotations

from pathlib import Path

import pytest

from veeksha.provenance import (
    capture_environment,
    file_digest,
    git_commit,
    python_environment,
)


@pytest.mark.unit
def test_capture_environment_shape() -> None:
    env = capture_environment()
    assert "veeksha" in env
    assert "version" in env["veeksha"]
    assert "git_commit" in env["veeksha"]
    assert "python" in env
    assert "version" in env["python"]
    assert "free_threaded" in env["python"]
    assert "platform" in env
    assert "packages" in env
    assert "transformers" in env["packages"]


@pytest.mark.unit
def test_file_digest_stable_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"abc")
    first = file_digest(path)
    second = file_digest(path)
    assert first is not None
    assert first == second
    assert first.startswith("sha256:")
    assert file_digest(tmp_path / "missing.bin") is None


@pytest.mark.unit
def test_file_digest_changes_when_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"v1")
    d1 = file_digest(path)
    path.write_bytes(b"v2-longer")
    d2 = file_digest(path)
    assert d1 != d2


@pytest.mark.unit
def test_git_commit_returns_hex_or_none() -> None:
    commit = git_commit()
    assert commit is None or (isinstance(commit, str) and len(commit) >= 7)


@pytest.mark.unit
def test_python_environment_reports_gil_state() -> None:
    info = python_environment()
    assert "gil_enabled" in info
    assert isinstance(info["free_threaded"], bool)
