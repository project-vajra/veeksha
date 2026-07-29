from typing import Optional

from vidhi import field, frozen_dataclass

from veeksha.cli.benchmark_command import BenchmarkCommand
from veeksha.config.client import (
    BaseClientConfig,
    OpenAIChatCompletionsClientConfig,
)
from veeksha.config.endpoint import EndpointConfig
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

# WandbConfig must be imported so vidhi registers it for polymorphic YAML
# deserialization. Removing this import will silently break wandb config loading.
from veeksha.config.wandb import WandbConfig  # noqa: F401


@frozen_dataclass
class BenchmarkConfig(BenchmarkCommand, name="run", default=True):
    """Run a benchmark against an LLM inference server.

    Invoked as ``veeksha benchmark run`` (or ``veeksha benchmark`` — run is the
    default subcommand).
    """

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
        help="Server configuration for managed servers. If set, it produces endpoint.",
    )
    endpoint: Optional[EndpointConfig] = field(
        None,
        help="Client-facing endpoint contract used to overwrite client.model, client.api_key and client.api_base.",
    )
    wandb: WandbConfig = field(
        default_factory=WandbConfig,
        help="Weights & Biases logging configuration.",
    )
    benchmark: Optional[str] = field(
        None,
        help=(
            "Named benchmark to fetch and run from the Hub "
            "(or a local definition directory). When set, the definition is "
            "frozen: only declared free variables (knobs) and the run target "
            "(endpoint/server, output_dir) may be set. Other config flags error "
            "unless --allow_config_override is true."
        ),
    )
    benchmark_revision: Optional[str] = field(
        None,
        help="Hub revision (tag or commit) for the named benchmark.",
    )
    allow_config_override: bool = field(
        False,
        help=(
            "When running a named benchmark, allow non-knob config overrides. "
            "Marks the run manifest as unpinned. Prefer declaring free "
            "variables in the definition instead."
        ),
    )
    allow_workload_drift: bool = field(
        False,
        help=(
            "When a named benchmark's computed workload fingerprint does not "
            "match its pins, log a warning instead of failing the run."
        ),
    )

    @classmethod
    def create_from_cli_args(cls):
        """Create BenchmarkConfig instances from CLI (including free variables).

        Returns:
            List of BenchmarkConfig instances (single or
            multiple configs if YAML expands to multiple configurations)
        """
        import sys

        from veeksha.cli.benchmark_run_cli import parse_benchmark_run_configs

        return parse_benchmark_run_configs(sys.argv[1:])

    def __post_init__(self):
        if not self.evaluators:
            raise ValueError("BenchmarkConfig.evaluators must be non-empty.")
        if self.server is not None and self.endpoint is not None:
            raise ValueError("BenchmarkConfig cannot set both server and endpoint")
        if self.benchmark_revision and not self.benchmark:
            raise ValueError(
                "benchmark_revision requires --benchmark to name the definition"
            )


# Runtime state attached to a frozen config with ``object.__setattr__`` rather
# than declared as fields: it is populated by the CLI, never parsed or
# serialized. ``dataclasses.replace`` only copies real fields, so any rebuild
# drops these unless they are carried over explicitly.
_SIDECAR_ATTRS = ("_named_benchmark_meta", "_knob_overrides", "_cli_provided_keys")


def carry_sidecar_attrs(source: object, target: object) -> object:
    """Copy non-field runtime attributes across a ``replace()`` rebuild.

    Losing ``_named_benchmark_meta`` silently disables the workload pin check —
    the run proceeds against an unverified workload — so every code path that
    rebuilds a :class:`BenchmarkConfig` must route through this.
    """
    for attr in _SIDECAR_ATTRS:
        value = getattr(source, attr, None)
        if value is not None:
            object.__setattr__(target, attr, value)
    return target
