from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from veeksha.core.audio_contract import (
    decode_pcm16_audio,
    measure_normalized_audio,
    measure_pcm16_audio,
)


def _pcm_bytes(samples: list[int]) -> bytes:
    return np.asarray(samples, dtype="<i2").tobytes()


def _wav_bytes(samples: list[int], sample_rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(_pcm_bytes(samples))
    return buffer.getvalue()


def test_measure_pcm16_audio_clean_fixture_reports_expected_metrics() -> None:
    audio = decode_pcm16_audio(
        _pcm_bytes([0, 8192, -8192, 16384, -16384]),
        raw_pcm=True,
        sample_rate=24000,
    )

    metrics = measure_pcm16_audio(audio)

    assert metrics.sample_count == 5
    assert metrics.peak_abs_amplitude == 0.5
    assert metrics.clipped_sample_fraction == 0.0
    assert metrics.rms == pytest.approx(np.sqrt(0.125))
    assert metrics.non_finite_sample_count == 0


def test_measure_pcm16_audio_clipped_fixture_counts_exact_int16_rails() -> None:
    audio = decode_pcm16_audio(
        _pcm_bytes([32767, -32768, 32766, -32767, 0]),
        raw_pcm=True,
        sample_rate=24000,
    )

    metrics = measure_pcm16_audio(audio)

    assert metrics.sample_count == 5
    assert metrics.peak_abs_amplitude == 1.0
    assert metrics.clipped_sample_fraction == 0.4
    assert metrics.non_finite_sample_count == 0


def test_decode_pcm16_audio_wav_and_raw_have_identical_samples() -> None:
    samples = [0, 8192, -8192, 16384, -16384]

    raw_audio = decode_pcm16_audio(_pcm_bytes(samples), raw_pcm=True, sample_rate=24000)
    wav_audio = decode_pcm16_audio(
        _wav_bytes(samples, sample_rate=22050),
        raw_pcm=False,
        sample_rate=24000,
    )

    np.testing.assert_array_equal(wav_audio.samples, raw_audio.samples)
    assert wav_audio.sample_rate == 22050
    assert measure_pcm16_audio(wav_audio) == measure_pcm16_audio(raw_audio)


def test_measure_normalized_audio_non_finite_fixture_is_counted() -> None:
    metrics = measure_normalized_audio(
        np.asarray([0.0, 0.5, np.nan, np.inf, -np.inf], dtype=np.float64)
    )

    assert metrics.sample_count == 5
    assert metrics.peak_abs_amplitude == 0.5
    assert metrics.clipped_sample_fraction == 0.0
    assert metrics.rms == pytest.approx(np.sqrt(0.125))
    assert metrics.non_finite_sample_count == 3


def test_decode_pcm16_audio_odd_length_payload_raises() -> None:
    with pytest.raises(ValueError, match="divisible by 2 bytes"):
        decode_pcm16_audio(b"\x00", raw_pcm=True, sample_rate=24000)


def test_decode_pcm16_audio_invalid_wav_raises() -> None:
    with pytest.raises(ValueError, match="Invalid WAV payload"):
        decode_pcm16_audio(b"not-a-wav", raw_pcm=False, sample_rate=24000)
