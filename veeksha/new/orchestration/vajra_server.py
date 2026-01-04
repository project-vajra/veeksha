from typing import List

from veeksha.logger import init_logger
from veeksha.new.config.server import VajraServerConfig
from veeksha.new.orchestration.server_manager import BaseServerManager

logger = init_logger(__name__)


class VajraServerManager(BaseServerManager):
    """Manager for Vajra inference servers."""

    def __init__(self, config: VajraServerConfig, output_dir: str):
        super().__init__(config, output_dir=output_dir)

        if config.engine.lower() != "vajra":
            logger.warning(
                f"VajraServerManager created with engine='{config.engine}'. "
                "Expected 'vajra'"
            )

    def _build_launch_command(self) -> List[str]:
        # Placeholder for Vajra command as it was not fully implemented in previous steps
        # Assuming similar structure
        command = [
            "python",
            "-m",
            "vajra_server.server",
            "--model",
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

        additional_args_dict = self.get_additional_args_dict()
        for key, value in additional_args_dict.items():
            if isinstance(value, bool) and value:
                command.append(f"--{key}")
            elif value is not None:
                command.extend([f"--{key}", str(value)])

        return command
