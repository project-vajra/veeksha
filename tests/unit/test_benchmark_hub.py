"""Tests for named-benchmark fetch/publish against the Hugging Face Hub.

``benchmark_hub`` imports ``huggingface_hub`` lazily inside each function, so
these patch the attributes on the real module rather than on ``hub`` itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from veeksha import benchmark_hub as hub


class _FakeHfApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def create_repo(self, repo_id, **kwargs) -> None:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))

    def upload_folder(self, **kwargs) -> None:
        self.calls.append(("upload_folder", kwargs))

    def create_tag(self, repo_id, **kwargs) -> None:
        self.calls.append(("create_tag", {"repo_id": repo_id, **kwargs}))

    def named(self, name: str) -> list[dict]:
        return [payload for call, payload in self.calls if call == name]


def _install_fake_api(monkeypatch: pytest.MonkeyPatch) -> _FakeHfApi:
    api = _FakeHfApi()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)
    return api


def _write_definition(local_dir: Path, name: str = "demo") -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "benchmark.yml").write_text(f"name: {name}\n", encoding="utf-8")
    (local_dir / "pins.json").write_text("{}\n", encoding="utf-8")
    return local_dir


def _fake_download_writing(name: str, *, calls: list[dict] | None = None):
    """Build a snapshot_download stub that materializes benchmarks/<name>."""

    def _download(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        staging = Path(kwargs["local_dir"])
        _write_definition(staging / "benchmarks" / name, name)
        return str(staging)

    return _download


def _staging_leftovers(root: Path) -> list[Path]:
    return [p for p in root.iterdir() if p.name.startswith(".hub_staging_")]


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hub_path_for_builds_a_scoped_prefix() -> None:
    assert hub.hub_path_for("synthetic-concurrency") == (
        "benchmarks/synthetic-concurrency"
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "stt/../evil"])
def test_hub_path_for_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError):
        hub.hub_path_for(name)


@pytest.mark.unit
def test_definition_dir_accepts_file_or_directory(tmp_path: Path) -> None:
    _write_definition(tmp_path)
    assert hub.definition_dir(tmp_path) == tmp_path
    assert hub.definition_dir(tmp_path / "benchmark.yml") == tmp_path


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fetch_benchmark_downloads_only_the_requested_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", _fake_download_writing("demo", calls=calls)
    )

    target = hub.fetch_benchmark("demo", repo="org/benches", local_dir=tmp_path)

    assert target == tmp_path / "benchmarks" / "demo"
    assert (target / "benchmark.yml").is_file()
    # A single benchmark must never pull the whole repository.
    assert calls[0]["allow_patterns"] == ["benchmarks/demo/**", "benchmarks/demo"]
    assert calls[0]["repo_id"] == "org/benches"
    assert calls[0]["repo_type"] == "dataset"
    assert _staging_leftovers(tmp_path) == []


@pytest.mark.unit
def test_fetch_benchmark_reuses_existing_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = _write_definition(tmp_path / "benchmarks" / "demo")
    (existing / "marker.txt").write_text("keep", encoding="utf-8")

    def _explode(**kwargs):
        raise AssertionError("snapshot_download should not be called")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _explode)

    target = hub.fetch_benchmark("demo", local_dir=tmp_path)

    assert (target / "marker.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.unit
def test_fetch_benchmark_force_replaces_existing_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = _write_definition(tmp_path / "benchmarks" / "demo")
    (existing / "stale.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", _fake_download_writing("demo")
    )

    target = hub.fetch_benchmark("demo", local_dir=tmp_path, force=True)

    assert (target / "benchmark.yml").is_file()
    assert not (target / "stale.txt").exists()
    assert _staging_leftovers(tmp_path) == []


@pytest.mark.unit
def test_fetch_benchmark_reports_a_missing_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _download_nothing(**kwargs):
        return kwargs["local_dir"]

    monkeypatch.setattr("huggingface_hub.snapshot_download", _download_nothing)

    with pytest.raises(FileNotFoundError, match="benchmarks/demo"):
        hub.fetch_benchmark("demo", repo="org/benches", local_dir=tmp_path)

    # Staging must be cleaned up even on the failure path.
    assert _staging_leftovers(tmp_path) == []


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_publish_benchmark_uploads_under_the_scoped_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _install_fake_api(monkeypatch)
    local = _write_definition(tmp_path / "demo")

    repo_id = hub.publish_benchmark(local, "demo", repo="org/benches")

    assert repo_id == "org/benches"
    upload = api.named("upload_folder")[0]
    assert upload["path_in_repo"] == "benchmarks/demo"
    assert upload["repo_type"] == "dataset"
    assert Path(upload["folder_path"]) == local
    assert api.named("create_repo")[0]["exist_ok"] is True
    assert api.named("create_tag") == []


@pytest.mark.unit
def test_publish_benchmark_tags_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _install_fake_api(monkeypatch)
    local = _write_definition(tmp_path / "demo")

    hub.publish_benchmark(local, "demo", repo="org/benches", tag="v1", private=True)

    assert api.named("create_tag")[0]["tag"] == "v1"
    assert api.named("create_repo")[0]["private"] is True


@pytest.mark.unit
def test_publish_benchmark_requires_a_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _install_fake_api(monkeypatch)

    with pytest.raises(FileNotFoundError, match="directory not found"):
        hub.publish_benchmark(tmp_path / "missing", "demo")

    bare = tmp_path / "demo"
    bare.mkdir()
    with pytest.raises(FileNotFoundError, match="benchmark.yml"):
        hub.publish_benchmark(bare, "demo")

    # Nothing may reach the Hub when validation fails.
    assert api.calls == []


# ---------------------------------------------------------------------------
# Definition loading
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_definition_reads_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.yml"
    path.write_text("name: demo\nversion: 1\n", encoding="utf-8")

    assert hub.load_definition(path)["name"] == "demo"


@pytest.mark.unit
def test_load_definition_rejects_a_non_mapping(tmp_path: Path) -> None:
    """A scalar document is rejected rather than surfacing as a definition.

    The loader itself raises here, so this pins the user-visible contract
    without asserting which layer produces the error.
    """
    path = tmp_path / "benchmark.yml"
    path.write_text("just a string\n", encoding="utf-8")

    with pytest.raises(ValueError):
        hub.load_definition(path)
