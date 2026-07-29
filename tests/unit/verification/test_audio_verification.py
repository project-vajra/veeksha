from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from veeksha.config.verification import (
    AudioVerificationConfig,
    UTMOSVerifierConfig,
    WERVerifierConfig,
    WhisperTranscriptionConfig,
)
from veeksha.verification import audio as audio_verification
from veeksha.verification.audio import (
    AudioVerificationError,
    LocalWhisperTranscriber,
    compute_wer,
    normalize_text,
    verify_audio_outputs,
)


def _wer_config(
    *, threshold: float = 0.3, fail_on_threshold: bool = False
) -> AudioVerificationConfig:
    return AudioVerificationConfig(
        fail_on_threshold=fail_on_threshold,
        wer=WERVerifierConfig(enabled=True, threshold=threshold),
    )


def _write_quality_fixture(output_dir: Path, references: list[str]) -> None:
    metrics_dir = output_dir / "metrics"
    audio_dir = output_dir / "audio_files"
    metrics_dir.mkdir(parents=True)
    audio_dir.mkdir()
    rows = [
        {"request_id": request_id, "input_text": reference}
        for request_id, reference in enumerate(references)
    ]
    (metrics_dir / "request_level_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    for request_id in range(len(references)):
        (audio_dir / f"request_{request_id}.wav").write_bytes(b"fixture")


@pytest.mark.unit
def test_seed_tts_style_normalization_and_wer_match_manual_edit_counts() -> None:
    assert normalize_text("Hello, ROCK'N'ROLL -- world!") == "hello rock'n'roll world"
    assert compute_wer("one two three", "one too three") == pytest.approx(1 / 3)
    assert compute_wer("one two three", "one three") == pytest.approx(1 / 3)
    assert compute_wer("one two three", "one two extra three") == pytest.approx(1 / 3)
    assert compute_wer("Hello, world!", "hello world") == 0.0
    assert compute_wer("", "") == 0.0
    assert compute_wer("", "unexpected") == 1.0


@pytest.mark.unit
def test_tts_verification_summary_matches_manual_percentiles(tmp_path: Path) -> None:
    references = ["one two three four"] * 4
    hypotheses = {
        "request_0.wav": "one two three four",
        "request_1.wav": "one two three",
        "request_2.wav": "one two",
        "request_3.wav": "",
    }
    _write_quality_fixture(tmp_path, references)

    summary = verify_audio_outputs(
        tmp_path,
        _wer_config(),
        transcribe_audio=lambda path: hypotheses[path.name],
    )

    assert summary.transcribed_requests == 4
    assert summary.passed_requests == 2
    assert summary.failed_requests == 2
    assert summary.error_requests == 0
    assert summary.wer_avg == pytest.approx(0.4375)
    assert summary.wer_p50 == pytest.approx(0.375)
    assert summary.wer_p90 == pytest.approx(0.85)
    assert summary.wer_p99 == pytest.approx(0.985)
    assert summary.wer_max == 1.0


@pytest.mark.unit
def test_strict_tts_verification_fails_closed_on_missing_audio(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "request_level_metrics.jsonl").write_text(
        json.dumps({"request_id": 7, "input_text": "hello"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AudioVerificationError, match="could not be verified"):
        verify_audio_outputs(
            tmp_path,
            _wer_config(fail_on_threshold=True),
            transcribe_audio=lambda _path: "hello",
        )


@pytest.mark.unit
def test_strict_tts_verification_fails_on_wer_threshold(tmp_path: Path) -> None:
    _write_quality_fixture(tmp_path, ["one two three"])

    with pytest.raises(AudioVerificationError, match="exceeded WER threshold"):
        verify_audio_outputs(
            tmp_path,
            _wer_config(threshold=0.2, fail_on_threshold=True),
            transcribe_audio=lambda _path: "one",
        )


@pytest.mark.unit
def test_utmos_scores_and_failures_are_counted_without_sample_bias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_quality_fixture(tmp_path, ["first", "second", "third"])
    scores = {
        "request_0.wav": 3.0,
        "request_1.wav": 4.0,
        "request_2.wav": None,
    }
    monkeypatch.setattr(
        audio_verification,
        "_utmos_predict_audio_path",
        lambda path, _config: scores[path.name],
    )
    config = AudioVerificationConfig(
        utmos=UTMOSVerifierConfig(enabled=True, device="cpu")
    )

    summary = verify_audio_outputs(tmp_path, config)

    assert summary.utmos_evaluated == 2
    assert summary.utmos_mean == 3.5
    assert summary.utmos_median == 3.5
    assert summary.utmos_failed == 1
    assert summary.error_requests == 1


@pytest.mark.unit
def test_whisper_judge_uses_pinned_language_task_and_beam() -> None:
    calls = []

    class FakeModel:
        def transcribe(self, path: str, **kwargs):
            calls.append((path, kwargs))
            return (
                iter([SimpleNamespace(text=" hello "), SimpleNamespace(text="world")]),
                None,
            )

    transcriber = object.__new__(LocalWhisperTranscriber)
    transcriber.config = AudioVerificationConfig(
        wer=WERVerifierConfig(
            enabled=True,
            whisper=WhisperTranscriptionConfig(language="en", beam_size=3),
        )
    )
    transcriber.model = FakeModel()

    assert transcriber.transcribe(Path("sample.wav")) == "hello world"
    assert calls == [
        (
            "sample.wav",
            {"language": "en", "task": "transcribe", "beam_size": 3},
        )
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"language": ""}, "language is required"),
        ({"beam_size": 0}, "beam_size must be >= 1"),
    ],
)
def test_whisper_judge_rejects_unusable_pinned_settings(
    kwargs: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        WhisperTranscriptionConfig(**kwargs)
