from dataclasses import field

from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.new.config.content import ContentConfig, SyntheticContentConfig

# from veeksha.new.config.server import ServerConfig
# from veeksha.new.config.client import ClientConfig


@frozen_dataclass(allow_from_file=True)
class BenchmarkConfig:
    seed: int = field(
        default=42, metadata={"help": "Seed for the random number generator."}
    )
    # runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    content: ContentConfig = field(
        default_factory=SyntheticContentConfig,
        metadata={"help": "The content configuration for the benchmark."},
    )
    # client: ClientConfig = field(default_factory=ClientConfig)
    # server: ServerConfig = field(default_factory=ServerConfig)
    # traffic: TrafficConfig = field(default_factory=RateTrafficConfig)
    # metrics: MetricsConfig = field(default_factory=MetricsConfig)
    # runtime: RuntimeConfig = field(default_factory=RuntimeConfig)  # threads, timeouts, telemetry

    # TODO: enable dashboard
    # dashboard: DashboardConfig = field(default_factory=DashboardConfig)

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

    def __post_init__(self):
        pass
