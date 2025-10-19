from dataclasses import field
from typing import Optional

from veeksha.config.client import ClientConfig
from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.dashboard import DashboardConfig
from veeksha.config.generators.request_generator.base_generator import (
    BaseRequestGeneratorConfig,
)
from veeksha.config.generators.request_generator.lmeval_generator import (
    LmevalRequestGeneratorConfig,
)
from veeksha.config.generators.request_generator.synthetic_generator import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.config.metrics import MetricsConfig
from veeksha.config.server import ServerConfig
from veeksha.config.utils import dataclass_to_dict
from veeksha.constants.configuration_constants import DEFAULT_SEED
from veeksha.logger import init_logger
from veeksha.types import RequestGeneratorType

logger = init_logger(__name__)


@frozen_dataclass(allow_from_file=True)
class BenchmarkConfig:
    """Configuration for LLM benchmarking.

    This configuration supports two modes of operation:

    1. **Managed Server Mode**: Provide `server_config` to automatically launch and
       manage an inference server (e.g., vLLM) before running the benchmark.

    2. **External Server Mode**: Leave `server_config` as None and specify `api_url`
       to connect to an already-running server.

    **Handling Field Redundancy**:

    When `server_config` is provided, warnings are issued for potential configuration issues:

    - `api_url`: Warns if at default value
    - `api_key`: Warns if at default value
    - `server_config.model`: Warns if it differs from `client_config.model`

    This design allows flexibility while alerting users to potential mismatches between
    the server being launched and the client making requests.
    """

    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": "Seed for the random number generator."},
    )
    timeout: int = field(
        default=1200,
        metadata={"help": "The amount of time to run the load test for."},
    )
    max_completed_requests: int = field(
        default=10,
        metadata={
            "help": "The number of requests to complete before finishing the test. Note "
            "that its possible for the test to timeout first."
        },
    )
    api_url: Optional[str] = field(
        default="http://localhost:8000/v1",
        metadata={"help": "The API URL for the benchmark."},
    )
    api_key: Optional[str] = field(
        default="token-abc123",
        metadata={"help": "The API key for the benchmark."},
    )
    client_config: ClientConfig = field(
        default_factory=ClientConfig,
        metadata={"help": "The client configuration for the benchmark."},
    )
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "The metrics configuration for the benchmark."},
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig,
        metadata={"help": "The request generator configuration for the benchmark."},
    )
    dashboard_config: DashboardConfig = field(
        default_factory=DashboardConfig, metadata={"help": "Dashboard configuration"}
    )
    server_config: Optional[ServerConfig] = field(
        default=None,
        metadata={
            "help": "Optional server configuration for automatic server management. "
            "If provided, the server will be launched before the benchmark. "
            "Warnings will be issued if api_url or api_key are at default values, "
            "or if server_config.model differs from client_config.model."
        },
    )
    runtime_telemetry_enabled: bool = field(
        default=False,
        metadata={
            "help": "Enable verbose dispatch runtime telemetry logs (backlog, prefetch rate)."
        },
    )
    num_prefetch_threads: int = field(
        default=4,
        metadata={
            "help": "Number of threads for prefetching/generating requests. "
            "Increase if request generation is a bottleneck."
        },
    )
    num_dispatcher_threads: int = field(
        default=4,
        metadata={
            "help": "Number of threads for dispatching requests to workers. "
            "Increase if dispatch scheduling is a bottleneck."
        },
    )
    num_results_processor_threads: int = field(
        default=4,
        metadata={
            "help": "Number of threads for processing completed requests. "
            "Increase if results aggregation is a bottleneck."
        },
    )
    num_request_runner_threads: int = field(
        default=10,
        metadata={
            "help": "Number of async worker threads for making concurrent requests. "
            "With GIL-free Python (python -Xgil=0), these threads run in true parallel. "
            "Each thread runs a uvloop event loop for handling concurrent HTTP requests."
        },
    )
    runtime_telemetry_enabled: bool = field(
        default=False,
        metadata={
            "help": "Enable verbose dispatch runtime telemetry logs (backlog, prefetch rate)."
        },
    )
    num_prefetch_threads: int = field(
        default=4,
        metadata={
            "help": "Number of threads for prefetching/generating requests. "
            "Increase if request generation is a bottleneck."
        },
    )
    num_dispatcher_threads: int = field(
        default=4,
        metadata={
            "help": "Number of threads for dispatching requests to workers. "
            "Increase if dispatch scheduling is a bottleneck."
        },
    )
    num_results_processor_threads: int = field(
        default=4,
        metadata={
            "help": "Number of threads for processing completed requests. "
            "Increase if results aggregation is a bottleneck."
        },
    )
    num_request_runner_threads: int = field(
        default=10,
        metadata={
            "help": "Number of async worker threads for making concurrent requests. "
            "With GIL-free Python (python -Xgil=0), these threads run in true parallel. "
            "Each thread runs a uvloop event loop for handling concurrent HTTP requests."
        },
    )

    def __post_init__(self):
        if self.num_request_runner_threads < 1:
            raise ValueError("num_request_runner_threads must be greater than 0")

        # Handle server_config if provided
        if self.server_config is not None:
            # Get the default values from the field definitions
            api_url_field = next(
                f for f in self.__dataclass_fields__.values() if f.name == "api_url"
            )
            api_key_field = next(
                f for f in self.__dataclass_fields__.values() if f.name == "api_key"
            )

            # Warn if api_url is at default value
            if self.api_url == api_url_field.default:
                logger.warning(
                    "api_url is at default value. Consider setting it explicitly when using server_config."
                )

            # Warn if api_key is at default value
            if self.api_key == api_key_field.default:
                logger.warning(
                    "api_key is at default value. Consider setting it explicitly when using server_config."
                )

            # Warn if server_config.model differs from client_config.model
            if self.server_config.model != self.client_config.model:
                logger.warning(
                    f"Client model differs from server. Setting server model from client_config: {self.client_config.model}"
                )
                object.__setattr__(
                    self.server_config, "model", self.client_config.model
                )

        if self.request_generator_config.get_type() == RequestGeneratorType.LMEVAL:
            logger.warning("Removing timeout for LMEval.")
            self.timeout = -1
            assert isinstance(
                self.request_generator_config, LmevalRequestGeneratorConfig
            )

            # Import the utility function locally to avoid circular imports
            from veeksha.generators.request_generator.lmeval_generator import (
                detect_task_types,
            )

            if detect_task_types(self.request_generator_config.tasks):
                object.__setattr__(self.client_config, "llm_api", "openai_completions")
                object.__setattr__(
                    self.client_config, "address_append_value", "completions"
                )
            else:
                object.__setattr__(self.client_config, "llm_api", "openai_chat")
                object.__setattr__(
                    self.client_config, "address_append_value", "chat/completions"
                )

    @classmethod
    def create_from_cli_args(cls):
        """Create BenchmarkConfig instances from CLI

        Returns:
            List of BenchmarkConfig instances (single or
            multiple configs if YAML expands to multiple configurations)
        """
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        instances = []
        for flat_config in flat_configs:
            instance = flat_config.reconstruct_original_dataclass()
            object.__setattr__(instance, "__flat_config__", flat_config)
            instances.append(instance)
        return instances

    def to_dict(self):
        flat_config = getattr(self, "__flat_config__", None)
        if flat_config is None:
            logger.debug("Flat config not found or is None. Using dataclass_to_dict.")
            return dataclass_to_dict(self)

        return self.__flat_config__.__dict__  # type: ignore
