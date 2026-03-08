from typing import Optional

from vidhi import field, frozen_dataclass, parse_cli_sweep

from veeksha.cli.base import VeekshaCommand
from veeksha.config.client import (
    BaseClientConfig,
    OpenAIChatCompletionsClientConfig,
)
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
from veeksha.config.wandb import WandbConfig  # noqa: F401


@frozen_dataclass
class BenchmarkConfig(VeekshaCommand, name="benchmark", default=True):
    """Run a benchmark against an LLM inference server."""

    output_dir: str = field(
        "benchmark_output",
        help="Base directory for all benchmark outputs (traces, metrics, logs)",
    )
    seed: int = field(42, help="Seed for the random number generator.")
    session_generator: BaseSessionGeneratorConfig = field(
        default_factory=SyntheticSessionGeneratorConfig,
        help="The session generator configuration for the benchmark.",
    )
    traffic_scheduler: BaseTrafficConfig = field(
        default_factory=RateTrafficConfig,
        help="The traffic scheduler configuration for the benchmark.",
    )
    evaluators: list[BaseEvaluatorConfig] = field(
        default_factory=lambda: [PerformanceEvaluatorConfig()],
        help="List of evaluators to run.",
    )
    client: BaseClientConfig = field(
        default_factory=OpenAIChatCompletionsClientConfig,
        help="The client configuration for the benchmark.",
    )
    runtime: RuntimeConfig = field(
        default_factory=RuntimeConfig,
        help="The runtime configuration for the benchmark.",
    )
    trace_recorder: TraceRecorderConfig = field(
        default_factory=TraceRecorderConfig,
        help="Trace recorder configuration. Records dispatched requests (unlike the evaluator, which records them after completion).",
    )
    server: Optional[BaseServerConfig] = field(
        None,
        help="Server configuration for managed servers. If set, client.model, client.api_key and client.api_base will be overwritten.",
    )
    wandb: WandbConfig = field(
        default_factory=WandbConfig,
        help="Weights & Biases logging configuration.",
    )

    @classmethod
    def create_from_cli_args(cls):
        """Create BenchmarkConfig instances from CLI

        Returns:
            List of BenchmarkConfig instances (single or
            multiple configs if YAML expands to multiple configurations)
        """
        return parse_cli_sweep(cls)

    def __post_init__(self):
        if not self.evaluators:
            raise ValueError("BenchmarkConfig.evaluators must be non-empty.")


