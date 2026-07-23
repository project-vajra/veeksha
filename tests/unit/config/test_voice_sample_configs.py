from pathlib import Path

import pytest
from vidhi import create_class_from_dict, load_yaml_config
from vidhi.utils import expand_dict

from veeksha.config.benchmark import BenchmarkConfig

SAMPLE_CONFIG_DIR = Path(__file__).resolve().parents[3] / "veeksha" / "sample_configs"


@pytest.mark.parametrize(
    ("filename", "client_type", "num_runs"),
    [
        ("stt_vajra.yml", "stt", 7),
        ("stt_vllm_realtime.yml", "stt", 7),
        ("stt_deepgram_flux.yml", "stt", 1),
        ("stt_deepgram_nova.yml", "stt", 1),
        ("stt_elevenlabs.yml", "stt", 1),
        ("tts_streaming_deepgram_flux.yml", "streaming_tts", 1),
        ("tts_streaming_deepgram_aura.yml", "streaming_tts", 1),
        ("tts_streaming_elevenlabs.yml", "streaming_tts", 2),
    ],
)
def test_voice_sample_config_deserializes(
    filename: str,
    client_type: str,
    num_runs: int,
) -> None:
    raw_config = load_yaml_config(str(SAMPLE_CONFIG_DIR / filename))
    expanded_configs = expand_dict(raw_config)

    assert len(expanded_configs) == num_runs
    benchmark_configs = [
        create_class_from_dict(BenchmarkConfig, config) for config in expanded_configs
    ]
    assert {str(config.client.get_type()) for config in benchmark_configs} == {
        client_type
    }


@pytest.mark.parametrize("filename", ["stt_vajra.yml", "stt_vllm_realtime.yml"])
def test_stt_sample_config_sweeps_canonical_concurrency(filename: str) -> None:
    raw_config = load_yaml_config(str(SAMPLE_CONFIG_DIR / filename))
    expanded_configs = expand_dict(raw_config)

    assert [
        config["traffic_scheduler"]["target_concurrent_sessions"]
        for config in expanded_configs
    ] == [1, 2, 4, 8, 16, 32, 64]
