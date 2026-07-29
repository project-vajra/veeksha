"""Synthetic audio channel generator.

Produces ``AudioChannelRequestContent`` pointing at a deterministic generated WAV
(a tone or silence). The clip is written once per parameter set to a cache
directory and reused across all requests, so a whole benchmark of STT sessions
decodes the same file once. Content is synthetic -- it exercises the audio-input
path (STT) without a real dataset.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Optional

from veeksha.config.generator.channel import AudioChannelGeneratorConfig
from veeksha.core.request_content import AudioChannelRequestContent
from veeksha.core.seeding import SeedManager
from veeksha.generator.channel.base import BaseChannelGenerator

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "veeksha_synthetic_audio")


def _clip_path(config: AudioChannelGeneratorConfig) -> str:
    name = (
        f"{config.waveform}_{config.duration_seconds:g}s_{config.sample_rate}hz"
        f"_{config.frequency_hz:g}.wav"
    )
    return os.path.join(_CACHE_DIR, name)


def _ensure_clip(config: AudioChannelGeneratorConfig) -> str:
    """Write the synthetic WAV once (idempotent) and return its path."""
    path = _clip_path(config)
    if os.path.exists(path):
        return path

    import numpy as np
    import soundfile as sf

    n = int(config.duration_seconds * config.sample_rate)
    if config.waveform == "silence":
        samples = np.zeros(n, dtype="float32")
    else:  # sine
        t = np.arange(n) / config.sample_rate
        samples = (0.1 * np.sin(2.0 * np.pi * config.frequency_hz * t)).astype(
            "float32"
        )

    os.makedirs(_CACHE_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    # format is explicit -- the .tmp name has no extension for soundfile to infer.
    sf.write(tmp, samples, config.sample_rate, subtype="PCM_16", format="WAV")
    os.replace(tmp, path)  # atomic; safe under the thundering herd at run start
    return path


class AudioChannelGenerator(BaseChannelGenerator):
    def __init__(
        self,
        config: AudioChannelGeneratorConfig,
        seed_manager: SeedManager,
        tokenizer_handle: Optional[Any] = None,
    ) -> None:
        super().__init__(config, seed_manager)
        self._config = config

    def generate_content(self, is_root: bool = False) -> AudioChannelRequestContent:
        return AudioChannelRequestContent(input_audio=_ensure_clip(self._config))
