from typing import List

from veeksha.config.server import VllmServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration.server_manager import BaseServerManager

logger = init_logger(__name__)


class VLLMServerManager(BaseServerManager):
    """Manager for vLLM inference servers."""

    def __init__(self, config: VllmServerConfig, output_dir: str):
        super().__init__(config, output_dir=output_dir)

    def _build_launch_command(self) -> List[str]:
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.config.model,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--api-key",
            self.config.api_key,
        ]

        if self.config.tensor_parallel_size > 1:
            command.extend(
                ["--tensor-parallel-size", str(self.config.tensor_parallel_size)]
            )

        if self.config.dtype:
            command.extend(["--dtype", self.config.dtype])

        if self.config.max_model_len is not None:
            command.extend(["--max-model-len", str(self.config.max_model_len)])

        additional_args_dict = self.get_additional_args_dict()
        special_keys = {"rope_scaling"}
        for key, value in additional_args_dict.items():
            if key in special_keys:
                continue
            if isinstance(value, bool):
                if value:
                    command.append(f"--{key}")
                continue
            if value is None:
                continue
            elif isinstance(value, (list, tuple)):
                command.append(f"--{key}")
                command.extend([str(v) for v in value])
            else:
                command.extend([f"--{key}", str(value)])

        command.extend(self._parse_additional_vllm_args())

        return command

    def _parse_additional_vllm_args(self) -> List[str]:
        args = []
        additional_args_dict = self.get_additional_args_dict()

        if "rope_scaling" in additional_args_dict:
            import json

            rope_config = additional_args_dict["rope_scaling"]
            rope_json = json.dumps(rope_config)
            args.extend(["--rope-scaling", rope_json])

        return args
