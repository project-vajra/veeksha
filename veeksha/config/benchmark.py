from dataclasses import field
from typing import Optional

from veeksha.config.client import (
    BaseClientConfig,
    OpenAIChatCompletionsClientConfig,
)
from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.evaluator import (
    BaseEvaluatorConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.config.generator.session import (
    BaseSessionGeneratorConfig,
    SyntheticSessionGeneratorConfig,
)
from veeksha.config.runtime import RuntimeConfig
from veeksha.config.server import BaseServerConfig
from veeksha.config.trace_recorder import TraceRecorderConfig
from veeksha.config.traffic import BaseTrafficConfig, RateTrafficConfig
from veeksha.config.wandb import WandbConfig
from veeksha.types import (
    ClientType,
    EvaluationType,
    SessionGeneratorType,
    TrafficType,
)


@frozen_dataclass(allow_from_file=True)
class BenchmarkConfig:
    output_dir: str = field(
        default="benchmark_output",
        metadata={
            "help": "Base directory for all benchmark outputs (traces, metrics, logs)"
        },
    )
    seed: int = field(
        default=42, metadata={"help": "Seed for the random number generator."}
    )
    session_generator: BaseSessionGeneratorConfig = field(
        default_factory=SyntheticSessionGeneratorConfig,
        metadata={
            "help": f"The session generator configuration for the benchmark. {SessionGeneratorType.help_str()}"
        },
    )
    traffic_scheduler: BaseTrafficConfig = field(
        default_factory=RateTrafficConfig,
        metadata={
            "help": f"The traffic scheduler configuration for the benchmark. {TrafficType.help_str()}"
        },
    )
    evaluators: list[BaseEvaluatorConfig] = field(
        default_factory=lambda: [PerformanceEvaluatorConfig()],
        metadata={"help": f"List of evaluators to run. {EvaluationType.help_str()}"},
    )
    client: BaseClientConfig = field(
        default_factory=OpenAIChatCompletionsClientConfig,
        metadata={
            "help": f"The client configuration for the benchmark. {ClientType.help_str()}"
        },
    )
    runtime: RuntimeConfig = field(
        default_factory=RuntimeConfig,
        metadata={"help": "The runtime configuration for the benchmark."},
    )
    trace_recorder: TraceRecorderConfig = field(
        default_factory=TraceRecorderConfig,
        metadata={
            "help": "Trace recorder configuration. Records dispatched requests (unlike the evaluator, which records them after completion)."
        },
    )
    server: Optional[BaseServerConfig] = field(
        default=None,
        metadata={
            "help": "Server configuration for managed servers. If set, client.model, client.api_key and client.api_base will be overwritten."
        },
    )
    wandb: WandbConfig = field(
        default_factory=WandbConfig,
        metadata={"help": "Weights & Biases logging configuration."},
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

    def __post_init__(self):
        if not self.evaluators:
            raise ValueError("BenchmarkConfig.evaluators must be non-empty.")
