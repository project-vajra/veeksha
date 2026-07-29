"""Unit tests for current engine-runner command construction."""

from textwrap import dedent

import pytest
import yaml
from vidhi import create_class_from_dict

from veeksha.config.server import BaseServerConfig, VllmServerConfig
from veeksha.orchestration.managed_engines import VllmOmniDockerRunner

pytestmark = pytest.mark.unit


def test_vllm_server_command_with_advanced_configuration():
    config = VllmServerConfig(
        hf_model="meta/test-model",
        model="served-model",
        deploy_config="/tmp/deploy.yaml",
        container_deploy_config="/etc/vllm/custom.yaml",
        container_port=9001,
        bootstrap="",
        engine_args=[
            "--max-num-batched-tokens",
            "4096",
            "--trust-remote-code",
        ],
    )

    command = VllmOmniDockerRunner(config, output_dir="/tmp")._build_server_cmd()

    assert command == [
        "vllm",
        "serve",
        "meta/test-model",
        "--omni",
        "--port",
        "9001",
        "--deploy-config",
        "/etc/vllm/custom.yaml",
        "--max-num-batched-tokens",
        "4096",
        "--trust-remote-code",
    ]


def test_server_config_loaded_from_yaml_builds_docker_command(tmp_path):
    deploy_config = tmp_path / "deploy.yaml"
    deploy_config.write_text("model: fixture\n")
    config_file = tmp_path / "server_config.yaml"
    config_file.write_text(
        dedent(
            f"""
            type: vllm
            hf_model: meta/demo-model
            model: served-model
            image: vllm-omni:test
            deploy_config: {deploy_config}
            docker_gpus: device=0,1
            bootstrap: ""
            env:
              TOKEN: fixture
            engine_args:
              - --max-num-batched-tokens
              - "512"
              - --trust-remote-code
            """
        ).strip()
    )

    server_config = create_class_from_dict(
        BaseServerConfig,
        yaml.safe_load(config_file.read_text()),
    )
    command = VllmOmniDockerRunner(
        server_config, output_dir=tmp_path
    )._build_docker_run_cmd()

    assert command[:3] == ["docker", "run", "-d"]
    assert command[command.index("--gpus") + 1] == "device=0,1"
    assert f"{deploy_config}:/etc/vllm-omni/{deploy_config.name}:ro" in command
    assert "TOKEN=fixture" in command
    assert "vllm-omni:test" in command
    assert command[-3:] == [
        "--max-num-batched-tokens",
        "512",
        "--trust-remote-code",
    ]
