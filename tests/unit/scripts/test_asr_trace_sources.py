import numpy as np
import pytest
import soundfile as sf

from scripts import prepare_audio_traces
from scripts.prepare_audio_traces import TraceSourceOptions, build_trace_source


@pytest.mark.unit
def test_aa_source_filters_complete_clips_by_max_duration(monkeypatch) -> None:
    samples = [
        {"transcript": "short clip", "url": "short"},
        {"transcript": "edge clip", "url": "edge"},
        {"transcript": "long clip", "url": "long"},
    ]
    durations = {"short": 10.0, "edge": 30.0, "long": 30.1}

    monkeypatch.setattr(
        prepare_audio_traces,
        "load_dataset",
        lambda *args, **kwargs: samples,
    )
    monkeypatch.setattr(
        prepare_audio_traces,
        "fetch_aa_audio",
        lambda row, repo: row["url"],
    )

    def decode_audio(source, options):
        return np.zeros(
            int(round(durations[source] * options.sample_rate)),
            dtype=np.float32,
        )

    monkeypatch.setattr(prepare_audio_traces, "decode_audio", decode_audio)

    source = build_trace_source(
        "aa_voxpopuli",
        TraceSourceOptions(),
    )
    clips = list(source.iter_clips())

    final_clips = list(
        prepare_audio_traces.finalize_clips(
            clips,
            max_duration_s=30.0,
            sample_rate=TraceSourceOptions().sample_rate,
        )
    )

    assert [clip.transcript for clip in final_clips] == ["short clip", "edge clip"]
    assert [clip.metadata["sample_id"] for clip in final_clips] == ["short", "edge"]


@pytest.mark.unit
def test_ami_word_timed_source_yields_relative_word_timestamps(tmp_path) -> None:
    audio_dir = tmp_path / "audio"
    words_dir = tmp_path / "words"
    audio_dir.mkdir()
    words_dir.mkdir()

    sf.write(audio_dir / "ES2001a.wav", np.zeros(16000, dtype=np.float32), 16000)
    (words_dir / "ES2001a.A.words.xml").write_text(
        """
        <root>
          <w starttime="0.10" endtime="0.30">hello</w>
          <w starttime="0.35" endtime="0.50">world</w>
        </root>
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(prepare_audio_traces.AMITraceSource, "AUDIO_DIR", str(audio_dir))
    monkeypatch.setattr(prepare_audio_traces.AMITraceSource, "WORDS_DIR", str(words_dir))

    source = build_trace_source(
        "ami_word_timed",
        TraceSourceOptions(min_duration_s=0.0, max_duration_s=2.0),
    )
    clips = list(source.iter_clips())

    assert len(clips) == 1
    clip = clips[0]
    assert clip.transcript == "hello world"
    assert clip.metadata["meeting_id"] == "ES2001a"
    assert clip.metadata["speaker_id"] == "A"
    assert clip.word_timestamps == [
        prepare_audio_traces.WordTiming(word="hello", start_ms=0.0, end_ms=200.0),
        prepare_audio_traces.WordTiming(word="world", start_ms=250.0, end_ms=400.0),
    ]
