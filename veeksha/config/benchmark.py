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
    
    When `server_config` is provided, the following fields are auto-populated:
    
    - `api_url`: Auto-populated from `server_config.get_api_base_url()` if not explicitly set
    - `api_key`: Auto-populated from `server_config.api_key` if not explicitly set
    - `server_config.model`: Auto-populated from `client_config.model` to avoid duplication
    
    This means you only need to specify the model once in `client_config.model`, and it will
    automatically be used for both the server launch and client requests.
    
    This design allows flexibility while preventing configuration mismatches between
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
    server_config: Optional[ServerConfig] = field(
        default=None,
        metadata={
            "help": "Optional server configuration for automatic server management. "
            "If provided, the server will be launched before the benchmark and "
            "api_url and api_key will be auto-populated if not explicitly set."
        },
    )

    def __post_init__(self):
        # Handle server_config if provided
        if self.server_config is not None:
            # Get the default values from the field definitions
            api_url_field = next(f for f in self.__dataclass_fields__.values() if f.name == 'api_url')
            api_key_field = next(f for f in self.__dataclass_fields__.values() if f.name == 'api_key')
            
            # Auto-populate api_url if not explicitly set (checking against default)
            if self.api_url == api_url_field.default:
                object.__setattr__(self, "api_url", self.server_config.get_api_base_url())
                logger.info(f"Auto-populated api_url from server_config: {self.api_url}")
            
            # Auto-populate api_key if not explicitly set (checking against default)
            if self.api_key == api_key_field.default:
                object.__setattr__(self, "api_key", self.server_config.api_key)
                logger.info("Auto-populated api_key from server_config")
            
            # Sync model from client_config to server_config to avoid user having to specify twice
            if self.server_config.model != self.client_config.model:
                logger.info(
                    f"Auto-populating server_config.model from client_config.model: {self.client_config.model}"
                )
                object.__setattr__(self.server_config, "model", self.client_config.model)
        
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
