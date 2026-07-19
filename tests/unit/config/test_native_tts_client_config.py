from __future__ import annotations

import pytest
from vidhi import create_class_from_dict

from veeksha.config.benchmark import BenchmarkConfig


@pytest.mark.parametrize(
    ("client_type", "model", "extra", "config_class"),
    [
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
