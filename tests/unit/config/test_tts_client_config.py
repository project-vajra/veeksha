from __future__ import annotations

import pytest
from vidhi import create_class_from_dict

from veeksha.client.registry import ClientRegistry
from veeksha.client.streaming_tts import StreamingTTSClient
from veeksha.client.tts import TTSClient
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import (
    StreamingTTSClientConfig,
    TTSClientConfig,
)
from veeksha.evaluator.performance.base import _STREAMING_TTS_CLIENT_TYPES
from veeksha.types import ClientType


@pytest.mark.parametrize(
    ("client_type", "provider", "model", "extra", "config_class"),
    [
        (
            "tts",
            "openai",
            "gpt-4o-mini-tts",
            {"voice_id": "alloy"},
            "TTSClientConfig",
        ),
        (
            "tts",
            "elevenlabs",
            "eleven_flash_v2_5",
            {"voice_id": "voice"},
            "TTSClientConfig",
        ),
        (
            "tts",
            "deepgram_flux",
            "flux-alexis-en",
            {},
            "TTSClientConfig",
        ),
        (
            "tts",
            "mistral",
            "voxtral-mini-tts-2603",
            {"voice_id": "voice"},
            "TTSClientConfig",
        ),
        (
            "streaming_tts",
            "openai_realtime",
            "realtime-tts",
            {},
            "StreamingTTSClientConfig",
        ),
        (
            "streaming_tts",
            "vajra",
            "vajra-tts",
            {"voice_id": "voice"},
            "StreamingTTSClientConfig",
        ),
        (
            "streaming_tts",
            "elevenlabs",
            "eleven_flash_v2_5",
            {"voice_id": "voice"},
            "StreamingTTSClientConfig",
        ),
        (
            "streaming_tts",
            "deepgram_flux",
            "flux-alexis-en",
            {},
            "StreamingTTSClientConfig",
        ),
        (
            "streaming_tts",
            "deepgram_aura",
            "aura-2-thalia-en",
            {},
            "StreamingTTSClientConfig",
        ),
        (
            "streaming_tts",
            "cartesia",
            "sonic-3.5",
            {"voice_id": "voice"},
            "StreamingTTSClientConfig",
        ),
    ],
)
def test_tts_provider_deserializes_through_transport_client_type(
    client_type: str,
    provider: str,
    model: str,
    extra: dict[str, str],
    config_class: str,
) -> None:
    config = create_class_from_dict(
        BenchmarkConfig,
        {
            "client": {
                "type": client_type,
                "provider": provider,
                "api_base": "https://provider.example",
                "model": model,
                **extra,
            }
        },
    )

    assert type(config.client).__name__ == config_class
    assert str(config.client.get_type()) == client_type
    assert config.client.provider == provider


def test_voice_client_types_are_transport_not_provider_specific() -> None:
    members = list(ClientType.__members__.values())

    assert len(members) == len({member.value for member in members})
    assert ClientType.TTS.value == 4
    assert ClientType.STREAMING_TTS.value == 5
    assert ClientType.STT.value == 6
    assert set(_STREAMING_TTS_CLIENT_TYPES) == {ClientType.STREAMING_TTS}


def test_registry_exposes_one_client_per_tts_transport() -> None:
    http_config = TTSClientConfig(
        provider="openai",
        api_base="http://localhost:8000",
        model="tts-model",
        voice_id="voice",
    )
    streaming_config = StreamingTTSClientConfig(
        provider="vajra",
        api_base="http://localhost:8000",
        model="streaming-tts-model",
    )

    http_client = ClientRegistry.get(ClientType.TTS, config=http_config)
    streaming_client = ClientRegistry.get(
        ClientType.STREAMING_TTS,
        config=streaming_config,
    )

    assert isinstance(http_client, TTSClient)
    assert isinstance(streaming_client, StreamingTTSClient)


@pytest.mark.parametrize("client_type", ["tts", "streaming_tts"])
def test_tts_configs_reject_unknown_provider(client_type: str) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_class_from_dict(
            BenchmarkConfig,
            {
                "client": {
                    "type": client_type,
                    "provider": "unknown",
                    "api_base": "https://provider.example",
                    "model": "model",
                    "voice_id": "voice",
                }
            },
        )


@pytest.mark.parametrize("max_buffer_delay_ms", [-1, 5001])
def test_cartesia_rejects_invalid_buffer_delay(max_buffer_delay_ms: int) -> None:
    with pytest.raises(ValueError, match="max_buffer_delay_ms"):
        StreamingTTSClientConfig(
            provider="cartesia",
            api_base="https://api.cartesia.ai",
            model="sonic-3.5",
            voice_id="voice",
            max_buffer_delay_ms=max_buffer_delay_ms,
        )
