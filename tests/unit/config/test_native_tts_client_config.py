from __future__ import annotations

import pytest
from vidhi import create_class_from_dict

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.evaluator.performance.base import _REALTIME_TTS_CLIENT_TYPES
from veeksha.types import ClientType


@pytest.mark.parametrize(
    ("client_type", "model", "extra", "config_class"),
    [
        (
            "vajra_tts_stream",
            "vajra-tts",
            {"voice_id": "voice"},
            "VajraTTSStreamClientConfig",
        ),
        (
            "elevenlabs_streaming_tts",
            "eleven_flash_v2_5",
            {"voice_id": "voice"},
            "ElevenLabsStreamingTTSClientConfig",
        ),
        (
            "deepgram_flux_streaming_tts",
            "flux-alexis-en",
            {},
            "DeepgramFluxStreamingTTSClientConfig",
        ),
        (
            "deepgram_aura_streaming_tts",
            "aura-2-thalia-en",
            {},
            "DeepgramAuraStreamingTTSClientConfig",
        ),
        (
            "elevenlabs_http_tts",
            "eleven_flash_v2_5",
            {"voice_id": "voice"},
            "ElevenLabsHTTPTTSClientConfig",
        ),
        (
            "deepgram_flux_http_tts",
            "flux-alexis-en",
            {},
            "DeepgramFluxHTTPClientConfig",
        ),
    ],
)
def test_native_tts_client_type_deserializes_from_public_name(
    client_type: str,
    model: str,
    extra: dict[str, str],
    config_class: str,
) -> None:
    config = create_class_from_dict(
        BenchmarkConfig,
        {
            "client": {
                "type": client_type,
                "api_base": "https://provider.example",
                "model": model,
                **extra,
            }
        },
    )

    assert type(config.client).__name__ == config_class
    assert str(config.client.get_type()) == client_type


def test_universal_voice_client_type_ids_are_unique() -> None:
    members = list(ClientType.__members__.values())

    assert len(members) == len({member.value for member in members})
    assert ClientType.VAJRA_TTS_STREAM.value == 7
    assert ClientType.DEEPGRAM_AURA_STREAMING_TTS.value == 12


def test_all_streaming_tts_clients_emit_completion_summaries() -> None:
    assert set(_REALTIME_TTS_CLIENT_TYPES) == {
        ClientType.REALTIME_TTS,
        ClientType.VAJRA_TTS_STREAM,
        ClientType.ELEVENLABS_STREAMING_TTS,
        ClientType.DEEPGRAM_FLUX_STREAMING_TTS,
        ClientType.DEEPGRAM_AURA_STREAMING_TTS,
    }
