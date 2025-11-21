"""Unit tests for engine-specific server managers with advanced configs."""

import json
from textwrap import dedent

import pytest  # type: ignore[import]
import yaml

from veeksha.config.server import ServerConfig
from veeksha.orchestration.vllm_server import VLLMServerManager
from veeksha.orchestration.vajra_server import VajraServerManager

pytestmark = pytest.mark.unit


def test_vllm_launch_command_with_advanced_configuration():
    config = ServerConfig(
        engine="vllm",
        model="meta/test-model",
        host="0.0.0.0",
        port=9001,
        api_key="secret-key",
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=8192,
        additional_args={
            "max-num-batched-tokens": 4096,
            "trust-remote-code": True,
            "disable-log-requests": False,
            "some_list": ["alpha", "beta"],
            "rope_scaling": {"type": "linear", "factor": 2.0},
        },
    )

    command = VLLMServerManager(config)._build_launch_command()

    # Base flags
    assert command[:3] == ["python", "-m", "vllm.entrypoints.openai.api_server"]
    assert command[command.index("--model") + 1] == "meta/test-model"
    assert command[command.index("--host") + 1] == "0.0.0.0"
    assert command[command.index("--port") + 1] == "9001"
    assert command[command.index("--api-key") + 1] == "secret-key"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--max-model-len") + 1] == "8192"

    # Additional args
    assert command[command.index("--max-num-batched-tokens") + 1] == "4096"
    assert "--trust-remote-code" in command
    assert "--disable-log-requests" not in command

    list_start = command.index("--some_list")
    assert command[list_start : list_start + 3] == ["--some_list", "alpha", "beta"]

    rope_index = command.index("--rope-scaling")
    assert command[rope_index + 1] == json.dumps({"type": "linear", "factor": 2.0})


def test_vajra_launch_command_with_extra_cli_options():
    config = ServerConfig(
        engine="vajra",
        model="meta/test-model",
        host="127.0.0.1",
        port=7777,
        api_key="another-secret",
        tensor_parallel_size=3,
        max_model_len=4096,
        additional_args={
            "pipeline_parallel_size": 4,
            "log_requests": True,
            "multi_value": ["foo", "bar"],
        },
    )

    command = VajraServerManager(config)._build_launch_command()

    assert command[:3] == ["python", "-m", "vajra.entrypoints.openai.server"]
    assert command[command.index("--model_deployment_config_model") + 1] == "meta/test-model"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "7777"
    assert command[command.index("--api_key") + 1] == "another-secret"
    assert command[command.index("--model_deployment_config_tensor_parallel_size") + 1] == "3"
    assert command[command.index("--model_deployment_config_max_model_len") + 1] == "4096"

    assert command[command.index("--pipeline_parallel_size") + 1] == "4"
    assert "--log_requests" in command

    mv_first = command.index("--multi_value")
    assert command[mv_first : mv_first + 2] == ["--multi_value", "foo"]
    assert command[mv_first + 2 : mv_first + 4] == ["--multi_value", "bar"]


def test_server_config_additional_args_loaded_from_yaml(tmp_path):
    config_file = tmp_path / "server_config.yaml"
    config_file.write_text(
        dedent(
            """
            engine: vllm
            model: meta/demo-model
            port: 8100
            additional_args:
              trust-remote-code: true
              max-num-batched-tokens: 512
              rope_scaling:
                type: linear
                factor: 1.25
            """
        ).strip()
    )

    data = yaml.safe_load(config_file.read_text())
    server_config = ServerConfig(**data)
    command = VLLMServerManager(server_config)._build_launch_command()

    assert "--trust-remote-code" in command
    idx = command.index("--max-num-batched-tokens")
    assert command[idx + 1] == "512"

    rope_idx = command.index("--rope-scaling")
    assert json.loads(command[rope_idx + 1]) == {"type": "linear", "factor": 1.25}
