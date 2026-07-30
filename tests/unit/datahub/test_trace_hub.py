"""Tests for the audio trace hub fetch/publish tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from datahub import trace_hub as hub
from datahub import prepare_audio_traces


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
                "sample_id": f"src-{index}",
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
        "public": False,
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
# Mixtures
# ---------------------------------------------------------------------------


def _write_pool(traces_root: Path, name: str, rows: int) -> Path:
    """A pool at traces_root/asr/<name> with unique sample ids."""
    pool_dir = traces_root / "asr" / name
    audio_dir = pool_dir / "audio" / name
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for index in range(rows):
        wav = audio_dir / f"clip_{index:05d}.wav"
        wav.write_bytes(b"RIFF")
        manifest_rows.append(
            {
                "session_id": index,
                "audio_file": wav.relative_to(pool_dir).as_posix(),
                "expected_transcript": f"{name} clip {index}",
                "sample_id": f"{name}-{index}",
            }
        )
    (pool_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    return pool_dir


def _mix_args(**overrides) -> argparse.Namespace:
    defaults = {
        "name": "mymix",
        "take": [],
        "reference": "",
        "sources": "",
        "seed": 42,
        "allow_duplicates": False,
        "force": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.unit
def test_mix_take_composes_pools(tmp_path: Path) -> None:
    _write_pool(tmp_path, "pool_a", 4)
    _write_pool(tmp_path, "pool_b", 3)

    hub.run_mix(_mix_args(take=["stt/pool_a:2", "stt/pool_b:1"]), traces_root=tmp_path)

    mix_dir = tmp_path / "asr" / "mymix"
    rows = hub.load_manifest_rows(mix_dir / "manifest.jsonl")
    assert len(rows) == 3
    assert [row["session_id"] for row in rows] == [0, 1, 2]
    assert all(row["audio_file"].startswith("../pool_") for row in rows)
    assert hub.validate_trace_dir(mix_dir) == []

    info = json.loads((mix_dir / "mixture.json").read_text())
    assert info["requires"] == ["stt/pool_a", "stt/pool_b"]
    assert info["recipe"]["seed"] == 42


@pytest.mark.unit
def test_mix_take_is_deterministic(tmp_path: Path) -> None:
    _write_pool(tmp_path, "pool_a", 10)
    args = _mix_args(take=["stt/pool_a:3"], force=True)

    hub.run_mix(args, traces_root=tmp_path)
    first = hub.load_manifest_rows(tmp_path / "asr" / "mymix" / "manifest.jsonl")
    hub.run_mix(args, traces_root=tmp_path)
    second = hub.load_manifest_rows(tmp_path / "asr" / "mymix" / "manifest.jsonl")

    assert [r["sample_id"] for r in first] == [r["sample_id"] for r in second]


@pytest.mark.unit
def test_mix_of_mixtures_flattens_requires_to_pools(tmp_path: Path) -> None:
    _write_pool(tmp_path, "pool_a", 4)
    _write_pool(tmp_path, "pool_b", 4)
    hub.run_mix(_mix_args(name="inner", take=["stt/pool_a:2"]), traces_root=tmp_path)

    hub.run_mix(
        _mix_args(name="outer", take=["stt/inner:1", "stt/pool_b:2"]),
        traces_root=tmp_path,
    )

    outer = tmp_path / "asr" / "outer"
    info = json.loads((outer / "mixture.json").read_text())
    # inner's dependency is flattened to its pool; inner itself is not required
    assert info["requires"] == ["stt/pool_a", "stt/pool_b"]
    rows = hub.load_manifest_rows(outer / "manifest.jsonl")
    assert rows[0]["audio_file"].startswith("../pool_a/")
    assert hub.validate_trace_dir(outer) == []


@pytest.mark.unit
def test_mix_dedupes_across_takes_and_errors_when_short(tmp_path: Path) -> None:
    _write_pool(tmp_path, "pool_a", 2)

    with pytest.raises(SystemExit, match="un-selected"):
        hub.run_mix(
            _mix_args(take=["stt/pool_a:2", "stt/pool_a:1"]), traces_root=tmp_path
        )

    hub.run_mix(
        _mix_args(take=["stt/pool_a:2", "stt/pool_a:1"], allow_duplicates=True),
        traces_root=tmp_path,
    )
    rows = hub.load_manifest_rows(tmp_path / "asr" / "mymix" / "manifest.jsonl")
    assert len(rows) == 3


@pytest.mark.unit
def test_mix_reference_mode_preserves_membership_and_order(tmp_path: Path) -> None:
    _write_pool(tmp_path, "pool_a", 3)
    _write_pool(tmp_path, "pool_b", 3)
    reference = tmp_path / "reference.jsonl"
    reference.write_text(
        "".join(
            json.dumps({"sample_id": sid, "audio_file": "x.wav"}) + "\n"
            for sid in ["pool_b-2", "pool_a-0", "pool_b-0"]
        ),
        encoding="utf-8",
    )

    hub.run_mix(
        _mix_args(reference=str(reference), sources="stt/pool_a,stt/pool_b"),
        traces_root=tmp_path,
    )

    rows = hub.load_manifest_rows(tmp_path / "asr" / "mymix" / "manifest.jsonl")
    assert [r["sample_id"] for r in rows] == ["pool_b-2", "pool_a-0", "pool_b-0"]


@pytest.mark.unit
def test_validate_rejects_paths_escaping_datasets_root(tmp_path: Path) -> None:
    mix_dir = tmp_path / "asr" / "evil"
    mix_dir.mkdir(parents=True)
    (mix_dir / "manifest.jsonl").write_text(
        json.dumps({"audio_file": "../../../../etc/passwd", "sample_id": "x"}) + "\n",
        encoding="utf-8",
    )
    problems = hub.validate_trace_dir(mix_dir)
    assert len(problems) == 1
    assert "escapes" in problems[0]


@pytest.mark.unit
def test_fetch_pulls_mixture_pool_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(**kwargs):
        staging = Path(kwargs["local_dir"])
        for pattern in kwargs["allow_patterns"]:
            repo_path = pattern.removesuffix("/**")
            prefix, name = repo_path.split("/")
            if name == "mymix":
                mix_dir = staging / repo_path
                mix_dir.mkdir(parents=True)
                (mix_dir / "manifest.jsonl").write_text(
                    json.dumps(
                        {
                            "session_id": 0,
                            "audio_file": "../pool_a/audio/pool_a/clip_00000.wav",
                            "sample_id": "pool_a-0",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (mix_dir / "mixture.json").write_text(
                    json.dumps({"requires": ["stt/pool_a"]}), encoding="utf-8"
                )
            else:
                # Serve the pool by building it at the staged repo path.
                pool_staging_root = staging / prefix
                pool_staging_root.mkdir(parents=True, exist_ok=True)
                pool = _write_pool(staging, name, rows=1)
                # _write_pool writes under asr/; move to the repo layout.
                target = pool_staging_root / name
                if not target.exists():
                    pool.rename(target)

    monkeypatch.setattr(hub, "snapshot_download", fake_download)

    hub.run_fetch(_fetch_args(datasets="stt/mymix"), traces_root=tmp_path)

    assert (tmp_path / "asr" / "mymix" / "manifest.jsonl").is_file()
    assert (tmp_path / "asr" / "pool_a" / "manifest.jsonl").is_file()
    assert hub.validate_trace_dir(tmp_path / "asr" / "mymix") == []


# ---------------------------------------------------------------------------
# Text (TTS) pools
# ---------------------------------------------------------------------------


def _write_text_pool(traces_root: Path, name: str, rows: int) -> Path:
    pool_dir = traces_root / "tts" / name
    pool_dir.mkdir(parents=True, exist_ok=True)
    (pool_dir / "manifest.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "session_id": i,
                    "sample_id": f"{name}-{i}",
                    "dataset": name,
                    "text": f"{name} sentence {i}",
                }
            )
            + "\n"
            for i in range(rows)
        ),
        encoding="utf-8",
    )
    return pool_dir


@pytest.mark.unit
def test_validate_accepts_text_rows_without_audio(tmp_path: Path) -> None:
    pool_dir = _write_text_pool(tmp_path, "seed_tts_en", 3)
    assert hub.validate_trace_dir(pool_dir) == []

    (pool_dir / "manifest.jsonl").write_text(
        json.dumps({"session_id": 0, "sample_id": "x"}) + "\n", encoding="utf-8"
    )
    problems = hub.validate_trace_dir(pool_dir)
    assert len(problems) == 1
    assert "neither audio_file nor text" in problems[0]


@pytest.mark.unit
def test_mix_composes_text_pools_without_path_rewriting(tmp_path: Path) -> None:
    _write_text_pool(tmp_path, "seed_tts_en", 5)
    _write_text_pool(tmp_path, "sharegpt", 5)

    hub.run_mix(
        _mix_args(name="tts_mix", take=["tts/seed_tts_en:2", "tts/sharegpt:3"]),
        traces_root=tmp_path,
    )

    mix_dir = tmp_path / "tts" / "tts_mix"
    rows = hub.load_manifest_rows(mix_dir / "manifest.jsonl")
    assert len(rows) == 5
    assert all("audio_file" not in row for row in rows)
    assert all(row["text"] for row in rows)
    assert hub.validate_trace_dir(mix_dir) == []
    info = json.loads((mix_dir / "mixture.json").read_text())
    assert info["requires"] == ["tts/seed_tts_en", "tts/sharegpt"]


@pytest.mark.unit
def test_sharegpt_flattening_extracts_assistant_turns() -> None:
    from datahub.prepare_text_traces import flatten_sharegpt_conversations

    conversations = [
        {
            "id": "conv1",
            "conversations": [
                {"from": "human", "value": "hi"},
                {"from": "gpt", "value": "hello there"},
                {"from": "human", "value": "more"},
                {"from": "gpt", "value": "  "},
                {"from": "gpt", "value": "second answer"},
            ],
        },
        {"id": "conv2", "conversations": [{"from": "human", "value": "only human"}]},
    ]

    rows = list(flatten_sharegpt_conversations(conversations))

    assert [row["sample_id"] for row in rows] == ["conv1:1", "conv1:4"]
    assert [row["text"] for row in rows] == ["hello there", "second answer"]
    assert [row["session_id"] for row in rows] == [0, 1]


# ---------------------------------------------------------------------------
# Variant derivation
# ---------------------------------------------------------------------------


def _variant_args(**overrides) -> argparse.Namespace:
    defaults = {
        "dataset": "stt/aa_public",
        "name": "bench100",
        "reference": "",
        "force": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_reference_manifest(path: Path, sample_ids: list[str]) -> None:
    rows = [
        {
            "session_id": index,
            "audio_file": f"audio/other/clip_{index:05d}.wav",
            "expected_transcript": f"clip {sample_id.split('-')[-1]}",
            "sample_id": sample_id,
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.unit
def test_variant_matches_reference_against_pool(tmp_path: Path) -> None:
    local_dir = tmp_path / "asr" / "aa_public"
    _write_trace_dir(local_dir, rows=4)
    reference = tmp_path / "reference.jsonl"
    _write_reference_manifest(reference, ["src-3", "src-1"])

    hub.run_variant(_variant_args(reference=str(reference)), traces_root=tmp_path)

    rows = hub.load_manifest_rows(local_dir / "manifest.bench100.jsonl")
    # Reference order preserved, session ids renumbered, pool audio reused.
    assert [row["sample_id"] for row in rows] == ["src-3", "src-1"]
    assert [row["session_id"] for row in rows] == [0, 1]
    assert rows[0]["audio_file"] == "audio/aa_voxpopuli/clip_00003.wav"
    assert hub.validate_trace_dir(local_dir) == []


@pytest.mark.unit
def test_variant_fails_when_reference_clip_missing_from_pool(
    tmp_path: Path,
) -> None:
    local_dir = tmp_path / "asr" / "aa_public"
    _write_trace_dir(local_dir, rows=2)
    reference = tmp_path / "reference.jsonl"
    _write_reference_manifest(reference, ["src-0", "src-9"])

    with pytest.raises(SystemExit, match="src-9"):
        hub.run_variant(_variant_args(reference=str(reference)), traces_root=tmp_path)
    assert not (local_dir / "manifest.bench100.jsonl").exists()


@pytest.mark.unit
def test_variant_refuses_overwrite_without_force(tmp_path: Path) -> None:
    local_dir = tmp_path / "asr" / "aa_public"
    _write_trace_dir(local_dir, rows=2)
    reference = tmp_path / "reference.jsonl"
    _write_reference_manifest(reference, ["src-0"])
    args = _variant_args(reference=str(reference))

    hub.run_variant(args, traces_root=tmp_path)
    with pytest.raises(SystemExit, match="--force"):
        hub.run_variant(args, traces_root=tmp_path)
    hub.run_variant(_variant_args(reference=str(reference), force=True), tmp_path)


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
    assert create_call["private"] is True
    (upload_call,) = api.named("upload_folder")
    assert upload_call["path_in_repo"] == "stt/aa_public"
    assert upload_call["folder_path"] == str(local_dir)
    assert "alignment/**" in upload_call["ignore_patterns"]
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
# Derived-build output directories in prepare_audio_traces
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_name_gives_derived_builds_their_own_directory() -> None:
    output_dir = prepare_audio_traces.output_dir_for_dataset_keys(
        ["aa_voxpopuli"], "aa_voxpopuli_tiled_5min"
    )
    assert (
        output_dir
        == prepare_audio_traces.TRACES_ROOT / "asr" / "aa_voxpopuli_tiled_5min"
    )

    # Without an explicit name, a single-source build is a source-pure pool
    # named after its dataset.
    assert (
        prepare_audio_traces.output_dir_for_dataset_keys(["aa_voxpopuli"])
        == prepare_audio_traces.TRACES_ROOT / "asr" / "aa_voxpopuli"
    )


@pytest.mark.unit
def test_multi_source_builds_require_explicit_output_name() -> None:
    with pytest.raises(SystemExit, match="--output-name"):
        prepare_audio_traces.output_dir_for_dataset_keys(
            ["aa_voxpopuli", "aa_earnings22"]
        )

    output_dir = prepare_audio_traces.output_dir_for_dataset_keys(
        ["aa_voxpopuli", "ami_word_timed"], "mixed_pool"
    )
    assert output_dir.name == "mixed_pool"


@pytest.mark.unit
def test_validate_args_rejects_bad_output_name() -> None:
    args = argparse.Namespace(
        clips_per_dataset=16,
        max_duration=30.0,
        target_duration=None,
        without_word_timestamping=False,
        datasets="aa_voxpopuli",
        output_name="../evil",
    )
    with pytest.raises(SystemExit, match="output-name"):
        prepare_audio_traces.validate_args(args)


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


# ---------------------------------------------------------------------------
# subset
# ---------------------------------------------------------------------------


def _subset_args(**overrides) -> argparse.Namespace:
    defaults = {
        "dataset": "stt/pool_a",
        "name": "pool_a_2",
        "count": 2,
        "seed": 42,
        "force": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.unit
def test_subset_materializes_its_own_audio(tmp_path: Path) -> None:
    """Unlike a mixture, a subset must stand alone: pool-relative paths and
    real audio files, so it can be published without its parent."""
    _write_pool(tmp_path, "pool_a", 10)

    hub.run_subset(_subset_args(count=3), traces_root=tmp_path)

    subset_dir = tmp_path / "asr" / "pool_a_2"
    rows = hub.load_manifest_rows(subset_dir / "manifest.jsonl")
    assert len(rows) == 3
    assert [row["session_id"] for row in rows] == [0, 1, 2]
    # Pool-relative, not sibling-relative like a mixture's rows.
    assert all(not row["audio_file"].startswith("..") for row in rows)
    assert all((subset_dir / row["audio_file"]).is_file() for row in rows)
    assert hub.validate_trace_dir(subset_dir) == []
    # A standalone pool, so publish's provenance check is satisfied.
    assert not (subset_dir / "mixture.json").exists()


@pytest.mark.unit
def test_subset_is_deterministic_and_ordered_by_parent(tmp_path: Path) -> None:
    _write_pool(tmp_path, "pool_a", 20)
    args = _subset_args(count=5, force=True)

    hub.run_subset(args, traces_root=tmp_path)
    first = hub.load_manifest_rows(tmp_path / "asr" / "pool_a_2" / "manifest.jsonl")
    hub.run_subset(args, traces_root=tmp_path)
    second = hub.load_manifest_rows(tmp_path / "asr" / "pool_a_2" / "manifest.jsonl")

    ids = [row["sample_id"] for row in first]
    assert ids == [row["sample_id"] for row in second]
    # Selection is what the seed fixes; order follows the parent manifest.
    parent = hub.load_manifest_rows(tmp_path / "asr" / "pool_a" / "manifest.jsonl")
    parent_order = [row["sample_id"] for row in parent]
    assert ids == [sid for sid in parent_order if sid in set(ids)]


@pytest.mark.unit
def test_subset_seed_changes_selection(tmp_path: Path) -> None:
    _write_pool(tmp_path, "pool_a", 50)

    hub.run_subset(_subset_args(count=10, seed=1), traces_root=tmp_path)
    one = hub.load_manifest_rows(tmp_path / "asr" / "pool_a_2" / "manifest.jsonl")
    hub.run_subset(_subset_args(count=10, seed=2, force=True), traces_root=tmp_path)
    two = hub.load_manifest_rows(tmp_path / "asr" / "pool_a_2" / "manifest.jsonl")

    assert [r["sample_id"] for r in one] != [r["sample_id"] for r in two]


@pytest.mark.unit
def test_subset_records_provenance_inheriting_parent(tmp_path: Path) -> None:
    pool_dir = _write_pool(tmp_path, "pool_a", 10)
    (pool_dir / "build_info.json").write_text(
        json.dumps({"source_revisions": {"pool_a": "rev-abc"}}), encoding="utf-8"
    )

    hub.run_subset(_subset_args(count=4, seed=7), traces_root=tmp_path)

    info = json.loads((tmp_path / "asr" / "pool_a_2" / "build_info.json").read_text())
    assert info["subset_of"] == "stt/pool_a"
    assert info["subset"] == {"count": 4, "seed": 7, "parent_clip_count": 10}
    # A subset introduces no new upstream source; it inherits the parent's.
    assert info["parent_build_info"]["source_revisions"] == {"pool_a": "rev-abc"}


@pytest.mark.unit
def test_subset_refuses_overselecting_and_existing_dir(tmp_path: Path) -> None:
    _write_pool(tmp_path, "pool_a", 3)

    with pytest.raises(SystemExit, match="only 3 clips"):
        hub.run_subset(_subset_args(count=4), traces_root=tmp_path)

    hub.run_subset(_subset_args(count=2), traces_root=tmp_path)
    with pytest.raises(SystemExit, match="pass --force"):
        hub.run_subset(_subset_args(count=2), traces_root=tmp_path)


@pytest.mark.unit
def test_subset_refuses_mixtures_and_self_overwrite(tmp_path: Path) -> None:
    _write_pool(tmp_path, "pool_a", 4)
    hub.run_mix(_mix_args(name="mymix", take=["stt/pool_a:2"]), traces_root=tmp_path)

    # A mixture's rows point at sibling audio; subsetting it would strand them.
    with pytest.raises(SystemExit, match="is a mixture"):
        hub.run_subset(
            _subset_args(dataset="stt/mymix", name="mymix_1", count=1),
            traces_root=tmp_path,
        )

    with pytest.raises(SystemExit, match="must differ"):
        hub.run_subset(
            _subset_args(dataset="stt/pool_a", name="pool_a", count=1),
            traces_root=tmp_path,
        )


@pytest.mark.unit
def test_subset_handles_text_pools(tmp_path: Path) -> None:
    _write_text_pool(tmp_path, "seed_tts_en", 8)

    hub.run_subset(
        _subset_args(dataset="tts/seed_tts_en", name="seed_tts_en_3", count=3),
        traces_root=tmp_path,
    )

    subset_dir = tmp_path / "tts" / "seed_tts_en_3"
    rows = hub.load_manifest_rows(subset_dir / "manifest.jsonl")
    assert len(rows) == 3
    assert all(row["text"] for row in rows)
    assert hub.validate_trace_dir(subset_dir) == []
