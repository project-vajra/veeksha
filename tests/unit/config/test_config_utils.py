from vidhi import create_class_from_dict

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.utils import redact_config_secrets


def test_redact_config_secrets_preserves_environment_references() -> None:
    config = {
        "client": {
            "api_key": "actual-secret",
            "api_key_env": "PROVIDER_API_KEY",
            "empty_secret": None,
        },
        "endpoint": {"authorization": "Bearer actual-secret"},
        "items": [{"password": "actual-secret", "model": "voice-model"}],
    }

    assert redact_config_secrets(config) == {
        "client": {
            "api_key": "<redacted>",
            "api_key_env": "PROVIDER_API_KEY",
            "empty_secret": None,
        },
        "endpoint": {"authorization": "<redacted>"},
        "items": [{"password": "<redacted>", "model": "voice-model"}],
    }


def test_redacted_env_backed_config_remains_parseable() -> None:
    config = {
        "client": {
            "type": "stt",
            "provider": "deepgram_nova",
            "api_base": "https://api.deepgram.com",
            "api_key": None,
            "api_key_env": "DEEPGRAM_API_KEY",
            "model": "nova-3",
        }
    }

    redacted = redact_config_secrets(config)
    parsed = create_class_from_dict(BenchmarkConfig, redacted)

    assert parsed.client.api_key is None
    assert parsed.client.api_key_env == "DEEPGRAM_API_KEY"
