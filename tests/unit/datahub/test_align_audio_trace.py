import json

import pytest

from datahub.prepare_audio_traces import (
    NEMO_MODEL,
    NeMoDockerAligner,
    NemoItem,
    read_nemo_word_timings,
)


@pytest.mark.unit
def test_attach_word_timings_maps_ctm_to_manifest_row(tmp_path) -> None:
    nemo_output_dir = tmp_path / "nemo_output"
    nemo_output_dir.mkdir()
    nemo_manifest = tmp_path / "nemo_manifest.jsonl"
    nemo_manifest.write_text("", encoding="utf-8")
    ctm_path = nemo_output_dir / "clip.ctm"
    ctm_path.write_text(
        "utt 1 0.100 0.200 hello\n" "utt 1 1.200 0.300 world\n",
        encoding="utf-8",
    )
    (nemo_output_dir / "nemo_manifest_with_output_file_paths.json").write_text(
        json.dumps({"word_ctm": str(ctm_path)}) + "\n",
        encoding="utf-8",
    )

    word_timings = read_nemo_word_timings(
        nemo_output_dir,
        nemo_manifest,
        [NemoItem(row_index=7, audio_path=tmp_path / "clip.wav", text="hello world")],
    )

    assert [(word.word, word.start_ms, word.end_ms) for word in word_timings[7]] == [
        ("hello", 100.0, 300.0),
        ("world", 1200.0, 1500.0),
    ]


@pytest.mark.unit
def test_nemo_forced_aligner_command_owns_docker_lifecycle(tmp_path) -> None:
    command = NeMoDockerAligner(
        image="nemo:test",
        gpus="device=0",
        cache_dir=tmp_path / "cache",
        extra_mounts=("/outside:/outside:ro",),
        repo_root=tmp_path,
    ).command(
        manifest_path=tmp_path / "manifest.jsonl",
        output_dir=tmp_path / "alignment" / "nemo_output",
        alignment_output_dir=tmp_path / "alignment",
    )

    assert command[:2] == ["docker", "run"]
    assert "--gpus" in command
    assert "device=0" in command
    assert "nemo:test" in command
    assert "/outside:/outside:ro" in command
    assert f"pretrained_name={NEMO_MODEL}" in command


@pytest.mark.unit
def test_parse_nemo_output_path_that_is_relative_to_working_directory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    nemo_output_dir = tmp_path / "alignment" / "nemo_output"
    nemo_output_dir.mkdir(parents=True)
    nemo_manifest = tmp_path / "alignment" / "nemo_manifest.jsonl"
    nemo_manifest.write_text("", encoding="utf-8")
    ctm_path = nemo_output_dir / "ctm" / "words" / "clip.ctm"
    ctm_path.parent.mkdir(parents=True)
    ctm_path.write_text("utt 1 0.100 0.200 hello NA lex NA\n", encoding="utf-8")
    (nemo_output_dir / "nemo_manifest_with_output_file_paths.json").write_text(
        json.dumps(
            {
                "words_level_ctm_filepath": ctm_path.relative_to(tmp_path).as_posix(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    word_timings = read_nemo_word_timings(
        nemo_output_dir,
        nemo_manifest,
        [NemoItem(row_index=0, audio_path=tmp_path / "clip.wav", text="hello")],
    )

    assert [(word.word, word.start_ms, word.end_ms) for word in word_timings[0]] == [
        ("hello", 100.0, 300.0),
    ]
