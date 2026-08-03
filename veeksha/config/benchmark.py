import csv
import json
from pathlib import Path
from typing import Optional

from vidhi import field, frozen_dataclass

from veeksha.cli.base import VeekshaCommand
from veeksha.cli.parsing import parse_cli_sweep
from veeksha.types import BenchmarkMode
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
class BenchmarkConfig(VeekshaCommand, name="benchmark", default=True):
    """Run a benchmark against an LLM inference server."""

    mode: BenchmarkMode = field(
        BenchmarkMode.PERFORMANCE,
        help=(
            "What the run measures. PERFORMANCE (default) holds offered load "
            "for a wall-clock window, wrapping the trace and discarding a "
            "warmup. QUALITY makes exactly one pass over the corpus with no "
            "warmup and no wrapping, so every clip is scored exactly once."
        ),
    )
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
        if self.mode is BenchmarkMode.QUALITY:
            self._validate_quality_mode()
        if self.server is not None and self.endpoint is not None:
            raise ValueError("BenchmarkConfig cannot set both server and endpoint")

    def _validate_quality_mode(self) -> None:
        """Reject configurations whose accuracy number would be meaningless.

        A quality run reports one WER over the corpus, so each clip must
        contribute exactly once. Wrapping re-scores whichever clips happen to
        come round again and silently weights them by resample count; a
        max_sessions above the corpus size does the same thing; one below it
        scores a prefix and reports it as if it were the corpus. None of these
        fail loudly on their own, which is why they are rejected here.
        """
        generator = self.session_generator
        wrap_mode = getattr(generator, "wrap_mode", None)
        if wrap_mode is None:
            raise ValueError(
                "QUALITY mode requires a trace-driven session generator with "
                f"wrap_mode; got {type(generator).__name__}, which cannot "
                "enumerate a finite corpus."
            )
        if wrap_mode:
            raise ValueError(
                "QUALITY mode requires session_generator.wrap_mode=False. "
                "Wrapping scores some clips more than once and weights them by "
                "resample count, so the reported WER is not the corpus WER."
            )
        corpus_size = self._quality_corpus_size(generator.trace_file)
        if self.runtime.max_sessions != corpus_size:
            raise ValueError(
                f"QUALITY mode requires runtime.max_sessions == corpus size "
                f"({corpus_size} sessions in {generator.trace_file}), got "
                f"{self.runtime.max_sessions}. Fewer scores a prefix and "
                f"reports it as the corpus WER; more (or -1) re-scores clips "
                f"and weights them by resample count. Set max_sessions: "
                f"{corpus_size}."
            )

    @staticmethod
    def _quality_corpus_size(trace_file: str) -> int:
        """Count the sessions in a trace so QUALITY can insist on one pass each.

        Counted at config time, not run time: a mismatch must fail before the
        server is touched, because a short run completes cleanly and reports a
        prefix WER as if it were the corpus WER.

        A session is a distinct session_id, not a line: the trace generator
        builds sessions with groupby("session_id"), and conversation-style
        traces carry several rows per session. Counting lines here would
        demand a max_sessions the generator can never produce — and the error
        message would confidently instruct the wrong value.
        """
        path = Path(trace_file)
        if not path.is_file():
            raise ValueError(
                f"QUALITY mode could not size the corpus: trace_file "
                f"{trace_file!r} does not exist (paths resolve against the "
                f"working directory)."
            )
        session_ids = set()
        if path.suffix.lower() == ".csv":
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if "session_id" not in row:
                        raise ValueError(
                            f"QUALITY mode could not size the corpus: "
                            f"{trace_file!r} has no session_id column, which "
                            f"the trace generator requires."
                        )
                    session_ids.add(row["session_id"])
        else:
            with path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if "session_id" not in record:
                        raise ValueError(
                            f"QUALITY mode could not size the corpus: a row of "
                            f"{trace_file!r} has no session_id field, which "
                            f"the trace generator requires."
                        )
                    session_ids.add(record["session_id"])
        if not session_ids:
            raise ValueError(
                f"QUALITY mode could not size the corpus: trace_file "
                f"{trace_file!r} contains no sessions."
            )
        return len(session_ids)
