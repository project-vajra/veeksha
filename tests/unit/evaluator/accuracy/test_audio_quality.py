from __future__ import annotations

import io
import wave

import pytest

from veeksha.evaluator.accuracy.audio import _make_wav_header


@pytest.mark.unit
def test_pcm_wav_header_preserves_wire_sample_rate_and_duration() -> None:
    sample_rate = 24000
    pcm = b"\x00\x00" * 480  # 20 ms of mono PCM16 at 24 kHz.
    wav = io.BytesIO(_make_wav_header(len(pcm), sample_rate) + pcm)

    with wave.open(wav, "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == sample_rate
        assert reader.getnframes() == 480
        assert reader.readframes(reader.getnframes()) == pcm
