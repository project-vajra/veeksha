from typing import List

from veeksha.config.server import SglangServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration.server_manager import BaseServerManager

logger = init_logger(__name__)


class SGLangServerManager(BaseServerManager):
    """Manager for SGLang inference servers."""

    def __init__(self, config: SglangServerConfig, output_dir: str):
        super().__init__(config, output_dir=output_dir)

    def _build_launch_command(self) -> List[str]:
        command = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.config.model,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
        ]

        if self.config.tensor_parallel_size > 1:
            command.extend(
                ["--tensor-parallel-size", str(self.config.tensor_parallel_size)]
            )

        if self.config.dtype:
            command.extend(["--dtype", self.config.dtype])

        if self.config.max_model_len is not None:
            command.extend(["--context-length", str(self.config.max_model_len)])

        additional_args_dict = self.get_additional_args_dict()
        for key, value in additional_args_dict.items():
            if isinstance(value, bool) and value:
                command.append(f"--{key}")
            elif value is None:
                continue
            elif isinstance(value, (list, tuple)):
                command.append(f"--{key}")
                command.append(f"[{', '.join([str(v) for v in value])}]")
            else:
                command.extend([f"--{key}", str(value)])

        command.extend(self._parse_additional_sglang_args())

        return command

    def _parse_additional_sglang_args(self) -> List[str]:
        args = []
        return args
