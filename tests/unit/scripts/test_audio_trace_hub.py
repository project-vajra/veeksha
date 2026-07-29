"""Tests for the audio trace hub fetch/publish tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import audio_trace_hub as hub
from scripts import prepare_audio_traces


def _write_trace_dir(local_dir: Path, *, rows: int = 2, variant: str = "") -> None:
    audio_dir = local_dir / "audio" / "aa_voxpopuli"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for index in range(rows):
        wav = audio_dir / f"clip_{index:05d}.wav"
        wav.write_bytes(b"RIFF")
        manifest_rows.append(
            {
                "session_id": index,
                "audio_file": wav.relative_to(local_dir).as_posix(),
                "expected_transcript": f"clip {index}",
            }
        )
    manifest = local_dir / hub.manifest_name(variant)
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )


def _fetch_args(**overrides) -> argparse.Namespace:
    defaults = {
        "repo": "org/veeksha-voice-traces",
        "datasets": "stt/aa_public",
        "revision": "",
        "variant": "",
        "force": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _publish_args(**overrides) -> argparse.Namespace:
    defaults = {
        "repo": "org/veeksha-voice-traces",
        "datasets": "stt/aa_public",
        "tag": "",
        "private": False,
        "commit_message": "",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


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


# ---------------------------------------------------------------------------
# Dataset spec resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_dataset_maps_stt_to_local_asr(tmp_path: Path) -> None:
    repo_path, local_dir = hub.resolve_dataset("stt/aa_public", tmp_path)
    assert repo_path == "stt/aa_public"
    assert local_dir == tmp_path / "asr" / "aa_public"

    repo_path, local_dir = hub.resolve_dataset("tts/seed_tts_en", tmp_path)
    assert repo_path == "tts/seed_tts_en"
    assert local_dir == tmp_path / "tts" / "seed_tts_en"


@pytest.mark.unit
@pytest.mark.parametrize(
    "spec",
    ["aa_public", "asr/aa_public", "stt/", "stt/../evil", "stt/a/b", "stt/.hidden"],
)
def test_resolve_dataset_rejects_bad_specs(spec: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        hub.resolve_dataset(spec, tmp_path)


@pytest.mark.unit
def test_manifest_name_variants() -> None:
    assert hub.manifest_name() == "manifest.jsonl"
    assert hub.manifest_name("max15s") == "manifest.max15s.jsonl"
    with pytest.raises(SystemExit):
        hub.manifest_name("../evil")


# ---------------------------------------------------------------------------
# Trace directory validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_trace_dir_passes_sound_directory(tmp_path: Path) -> None:
    _write_trace_dir(tmp_path)
    assert hub.validate_trace_dir(tmp_path) == []


@pytest.mark.unit
def test_validate_trace_dir_reports_problems(tmp_path: Path) -> None:
    assert hub.validate_trace_dir(tmp_path) == [
        f"{tmp_path / 'manifest.jsonl'} is missing"
    ]

    _write_trace_dir(tmp_path)
    (tmp_path / "audio" / "aa_voxpopuli" / "clip_00001.wav").unlink()
    problems = hub.validate_trace_dir(tmp_path)
    assert len(problems) == 1
    assert "clip_00001.wav" in problems[0]


@pytest.mark.unit
def test_validate_trace_dir_covers_variant_manifests(tmp_path: Path) -> None:
    _write_trace_dir(tmp_path)
    (tmp_path / "manifest.broken.jsonl").write_text(
        json.dumps({"audio_file": "audio/missing.wav"}) + "\n",
        encoding="utf-8",
    )
    problems = hub.validate_trace_dir(tmp_path)
    assert len(problems) == 1
    assert problems[0].startswith("manifest.broken.jsonl:1")


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _fake_snapshot_download(**kwargs):
    staging = Path(kwargs["local_dir"])
    for pattern in kwargs["allow_patterns"]:
        repo_path = pattern.removesuffix("/**")
        _write_trace_dir(staging / repo_path)


@pytest.mark.unit
def test_fetch_moves_snapshot_into_local_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hub, "snapshot_download", _fake_snapshot_download)

    hub.run_fetch(_fetch_args(), traces_root=tmp_path)

    local_dir = tmp_path / "asr" / "aa_public"
    assert (local_dir / "manifest.jsonl").is_file()
    assert (local_dir / "audio" / "aa_voxpopuli" / "clip_00000.wav").is_file()
    assert not list(tmp_path.glob(".hub_staging_*"))


@pytest.mark.unit
def test_fetch_refuses_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hub, "snapshot_download", _fake_snapshot_download)
    existing = tmp_path / "asr" / "aa_public"
    existing.mkdir(parents=True)
    (existing / "sentinel").touch()

    with pytest.raises(SystemExit, match="--force"):
        hub.run_fetch(_fetch_args(), traces_root=tmp_path)
    assert (existing / "sentinel").exists()

    hub.run_fetch(_fetch_args(force=True), traces_root=tmp_path)
    assert not (existing / "sentinel").exists()
    assert (existing / "manifest.jsonl").is_file()


@pytest.mark.unit
def test_fetch_missing_variant_errors_with_available_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(**kwargs):
        staging = Path(kwargs["local_dir"])
        _write_trace_dir(staging / "stt" / "aa_public")
        _write_trace_dir(staging / "stt" / "aa_public", variant="max15s")

    monkeypatch.setattr(hub, "snapshot_download", fake_download)

    with pytest.raises(SystemExit, match="max15s"):
        hub.run_fetch(_fetch_args(variant="nope"), traces_root=tmp_path)


@pytest.mark.unit
def test_fetch_requires_repo(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--repo"):
        hub.run_fetch(_fetch_args(repo=""), traces_root=tmp_path)


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_publish_uploads_validated_trace_and_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _FakeHfApi()
    monkeypatch.setattr(hub, "HfApi", lambda: api)
    local_dir = tmp_path / "asr" / "aa_public"
    _write_trace_dir(local_dir)

    hub.run_publish(_publish_args(tag="v1"), traces_root=tmp_path)

    (create_call,) = api.named("create_repo")
    assert create_call["exist_ok"] is True
    (upload_call,) = api.named("upload_folder")
    assert upload_call["path_in_repo"] == "stt/aa_public"
    assert upload_call["folder_path"] == str(local_dir)
    (tag_call,) = api.named("create_tag")
    assert tag_call["tag"] == "v1"

    # Publish-time provenance is created when the build did not record any.
    build_info = json.loads((local_dir / "build_info.json").read_text())
    assert "publish time" in build_info["note"]


@pytest.mark.unit
def test_publish_refuses_invalid_trace_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _FakeHfApi()
    monkeypatch.setattr(hub, "HfApi", lambda: api)
    local_dir = tmp_path / "asr" / "aa_public"
    _write_trace_dir(local_dir)
    (local_dir / "audio" / "aa_voxpopuli" / "clip_00000.wav").unlink()

    with pytest.raises(SystemExit, match="Refusing to publish"):
        hub.run_publish(_publish_args(), traces_root=tmp_path)
    assert api.calls == []


# ---------------------------------------------------------------------------
# Build provenance in prepare_audio_traces
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_build_info_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prepare_audio_traces, "git_commit", lambda: "abc123")
    monkeypatch.setattr(
        prepare_audio_traces,
        "resolve_source_revisions",
        lambda keys: {key: "rev" for key in keys},
    )
    args = argparse.Namespace(
        clips_per_dataset=16,
        max_duration=30.0,
        target_duration=None,
        without_word_timestamping=False,
        datasets="aa_voxpopuli",
    )

    path = prepare_audio_traces.write_build_info(
        output_dir=tmp_path,
        args=args,
        dataset_keys=["aa_voxpopuli"],
        clip_count=16,
    )

    info = json.loads(path.read_text())
    assert info["veeksha_git_commit"] == "abc123"
    assert info["source_revisions"] == {"aa_voxpopuli": "rev"}
    assert info["clip_count"] == 16
    assert info["word_timestamping"]["nemo_model"] == prepare_audio_traces.NEMO_MODEL
    assert info["seed"] == prepare_audio_traces.DEFAULT_SEED


@pytest.mark.unit
def test_write_build_info_notes_disabled_timestamping(tmp_path: Path) -> None:
    args = argparse.Namespace(
        clips_per_dataset=0,
        max_duration=None,
        target_duration=None,
        without_word_timestamping=True,
        datasets="ami_word_timed",
    )

    path = prepare_audio_traces.write_build_info(
        output_dir=tmp_path,
        args=args,
        dataset_keys=["ami_word_timed"],
        clip_count=3,
    )

    info = json.loads(path.read_text())
    assert info["word_timestamping"] is None
    assert info["source_revisions"] == {
        "ami_word_timed": prepare_audio_traces.AMI_ANNOTATIONS_ARCHIVE
    }
