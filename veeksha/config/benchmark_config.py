from dataclasses import field
from typing import Optional

from veeksha.config.api_client_config import BaseApiClientConfig, OpenAIChatApiClientConfig
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
    num_api_clients: int = field(
        default=2,
        metadata={"help": "The number of clients to use for benchmark."},
    )
    api_client_config: BaseApiClientConfig = field(
        default_factory=OpenAIChatApiClientConfig,
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
