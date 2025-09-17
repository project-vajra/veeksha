from dataclasses import field
from typing import Optional

from veeksha.config.client_config import ClientConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.core.base_entrypoint_config import BaseEntrypointConfig
from veeksha.config.generators.request_generator.base_generator_config import (
    BaseRequestGeneratorConfig,
)
from veeksha.config.generators.request_generator.lmeval_generator_config import (
    LmevalRequestGeneratorConfig,
)
from veeksha.config.generators.request_generator.synthetic_generator_config import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.config.metrics_config import MetricsConfig
from veeksha.constants.configuration_constants import DEFAULT_SEED
from veeksha.logger import init_logger
from veeksha.types import RequestGeneratorType

logger = init_logger(__name__)


@frozen_dataclass(allow_from_file=True)
class BenchmarkConfig(BaseEntrypointConfig):
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

    def __post_init__(self):
        if self.request_generator_config.get_type() == RequestGeneratorType.LMEVAL:
            logger.warning("Removing timeout for LMEval.")
            self.timeout = -1
            assert isinstance(
                self.request_generator_config, LmevalRequestGeneratorConfig
            )

            # Import the utility function locally to avoid circular imports
            from veeksha.generators.request_generator.lmeval_request_generator import (
                requires_logits,
            )

            if requires_logits(self.request_generator_config.tasks):
                assert self.client_config.llm_api == "openai_completions", "llm_api must be openai_completions"
