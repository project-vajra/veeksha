from pathlib import Path

import pytest
from vidhi import create_class_from_dict, load_yaml_config
from vidhi.utils import expand_dict

from veeksha.config.benchmark import BenchmarkConfig

SAMPLE_CONFIG_DIR = Path(__file__).resolve().parents[3] / "veeksha" / "sample_configs"

REALTIME_TTS_CONFIGS = [
    (
        "tts_streaming_deepgram_flux_haley_seed_tts.yml",
        "deepgram_flux",
        "flux-haley-en",
        "seed_tts_text",
    ),
    (
        "tts_streaming_deepgram_flux_haley_sharegpt.yml",
        "deepgram_flux",
        "flux-haley-en",
        "sharegpt",
    ),
    (
        "tts_streaming_deepgram_aura_2_thalia_seed_tts.yml",
        "deepgram_aura",
        "aura-2-thalia-en",
        "seed_tts_text",
    ),
    (
        "tts_streaming_deepgram_aura_2_thalia_sharegpt.yml",
        "deepgram_aura",
        "aura-2-thalia-en",
        "sharegpt",
    ),
    (
        "tts_streaming_elevenlabs_multilingual_v2_seed_tts.yml",
        "elevenlabs",
        "eleven_multilingual_v2",
        "seed_tts_text",
    ),
    (
        "tts_streaming_elevenlabs_multilingual_v2_sharegpt.yml",
        "elevenlabs",
        "eleven_multilingual_v2",
        "sharegpt",
    ),
    (
        "tts_streaming_elevenlabs_flash_v2_5_seed_tts.yml",
        "elevenlabs",
        "eleven_flash_v2_5",
        "seed_tts_text",
    ),
    (
        "tts_streaming_elevenlabs_flash_v2_5_sharegpt.yml",
        "elevenlabs",
        "eleven_flash_v2_5",
        "sharegpt",
    ),
    (
        "tts_streaming_cartesia_sonic_3_5_seed_tts.yml",
        "cartesia",
        "sonic-3.5",
        "seed_tts_text",
    ),
    (
        "tts_streaming_cartesia_sonic_3_5_sharegpt.yml",
        "cartesia",
        "sonic-3.5",
        "sharegpt",
    ),
]


@pytest.mark.parametrize(
    ("filename", "provider", "model", "trace_flavor"), REALTIME_TTS_CONFIGS
)
def test_realtime_tts_sample_config_deserializes_with_common_workload(
    filename: str,
    provider: str,
    model: str,
    trace_flavor: str,
) -> None:
    raw_config = load_yaml_config(str(SAMPLE_CONFIG_DIR / filename))
    expanded_configs = expand_dict(raw_config)

    assert len(expanded_configs) == 1
    config = create_class_from_dict(BenchmarkConfig, expanded_configs[0])

    assert str(config.client.get_type()) == "streaming_tts"
    assert config.client.provider == provider
    assert config.client.model == model
    assert config.client.pacing.tokens_per_second == 50
    assert config.client.pacing.tokens_per_delta == 1
    assert config.client.pacing.gap_distribution == "fixed"
    assert str(config.session_generator.flavor.get_type()) == trace_flavor
    assert config.session_generator.flavor.min_chars == 20
    assert config.session_generator.flavor.max_chars == 500
    assert config.traffic_scheduler.target_concurrent_sessions == 1
    trace_dir = "seed_tts" if trace_flavor == "seed_tts_text" else "sharegpt"
    assert config.output_dir == f"benchmark_output/tts_streaming_{model}_{trace_dir}"
    assert config.runtime.max_sessions == 100
    if provider == "cartesia":
        assert config.client.voice_id == "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"
        assert config.client.max_buffer_delay_ms == 3000

    performance = config.evaluators[0]
    assert performance.audio_channel.fluidity_attribution_mode == "conservative"
    assert performance.audio_channel.persist_raw_timing is True
    assert [slo.metric for slo in performance.slos] == [
        "first_input_to_first_audio_ms",
        "trigger_to_first_playable_audio_ms",
        "ttfc",
        "rtf",
        "user_audio_fluidity_index",
    ]

    assert len(config.evaluators) == 2
    audio_output = config.evaluators[1]
    assert audio_output.save_audio_files is True
    assert audio_output.verification.is_enabled() is False
