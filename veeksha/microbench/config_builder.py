"""Build full BenchmarkConfig instances from simplified MicrobenchmarkConfig."""

import math

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import OpenAIChatCompletionsClientConfig
from veeksha.config.evaluator import (
    DecodeWindowConfig,
    PerformanceEvaluatorConfig,
    TextChannelPerformanceConfig,
)
from veeksha.config.generator.channel import TextChannelGeneratorConfig
from veeksha.config.generator.length import (
    FixedLengthGeneratorConfig,
    StairLengthGeneratorConfig,
)
from veeksha.config.generator.requested_output import (
    OutputSpecConfig,
    TextOutputSpecConfig,
)
from veeksha.config.generator.session import SyntheticSessionGeneratorConfig
from veeksha.config.generator.session_graph import (
    SingleRequestSessionGraphGeneratorConfig,
)
from veeksha.config.runtime import RuntimeConfig
from veeksha.config.trace_recorder import TraceRecorderConfig
from veeksha.config.traffic import (
    ConcurrentTrafficConfig,
    SequentialLaunchTrafficConfig,
)
from veeksha.microbench.config import MicrobenchmarkConfig

_OUTPUT_TOKEN_MULTIPLIER = 2


def build_benchmark_configs(cfg: MicrobenchmarkConfig) -> list[BenchmarkConfig]:
    """Convert a simplified microbenchmark config into a list of full BenchmarkConfigs."""
    if cfg.type == "prefill":
        return _build_prefill_benchmark(cfg)
    elif cfg.type == "decode":
        return _build_decode_benchmarks(cfg)
    else:
        raise ValueError(f"Unknown config type: {cfg.type}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_client_config(
    cfg: MicrobenchmarkConfig,
) -> OpenAIChatCompletionsClientConfig:
    return OpenAIChatCompletionsClientConfig(
        model=cfg.model,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        request_timeout=cfg.request_timeout,
        max_tokens_param=cfg.max_tokens_param,
        ignore_eos=cfg.ignore_eos,
    )


def compute_prefill_iterations(
    input_length: int, chunk_size: int, active_decodes: int
) -> int:
    """Iterations needed to prefill one request given active decode slots.

    Each iteration has a token budget of *chunk_size*.  Active decode
    requests consume one token each, leaving the rest for prefill.
    """
    effective_chunk = chunk_size - active_decodes
    assert (
        effective_chunk > 0
    ), f"chunk_size ({chunk_size}) must exceed active_decodes ({active_decodes})"
    return math.ceil(input_length / effective_chunk)


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


def _build_prefill_benchmark(cfg: MicrobenchmarkConfig) -> list[BenchmarkConfig]:
    total_sessions = len(cfg.input_lengths) * cfg.samples_per_length
    return [
        BenchmarkConfig(
            output_dir=cfg.output_dir,
            seed=cfg.seed,
            session_generator=SyntheticSessionGeneratorConfig(
                session_graph=SingleRequestSessionGraphGeneratorConfig(),
                channels=[
                    TextChannelGeneratorConfig(
                        body_length_generator=StairLengthGeneratorConfig(
                            values=cfg.input_lengths,
                            repeat_each=cfg.samples_per_length,
                            wrap=False,
                        ),
                        shared_prefix_ratio=0.0,
                    )
                ],
                output_spec=OutputSpecConfig(
                    text=TextOutputSpecConfig(
                        output_length_generator=FixedLengthGeneratorConfig(
                            value=cfg.output_tokens,
                        ),
                    ),
                ),
            ),
            traffic_scheduler=ConcurrentTrafficConfig(
                target_concurrent_sessions=1,
                rampup_seconds=0,
                cancel_session_on_failure=False,
            ),
            evaluators=[
                PerformanceEvaluatorConfig(
                    stream_metrics=False,
                ),
            ],
            client=_build_client_config(cfg),
            runtime=RuntimeConfig(
                max_sessions=total_sessions,
                benchmark_timeout=cfg.benchmark_timeout,
                pregenerate_sessions=True,
            ),
            trace_recorder=TraceRecorderConfig(enabled=False),
        )
    ]


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def required_decode_output_tokens(
    samples_per_length: int,
    batch_size: int,
    input_length: int,
    chunk_size: int,
) -> int:
    """Compute output tokens for a decode benchmark run.

    Request 0 enters decode first and must still be generating when the
    last request finishes prefilling, plus *samples_per_length* additional
    pure-decode iterations for measurement.

        output_tokens = samples_per_length
            + (batch_size - 1) * ceil(input_length / (chunk_size - batch_size))
    """
    if batch_size == 1:
        return samples_per_length
    ramp_up = (batch_size - 1) * compute_prefill_iterations(
        input_length, chunk_size, batch_size
    )
    return samples_per_length + ramp_up


def _build_decode_benchmarks(cfg: MicrobenchmarkConfig) -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    for batch_size in cfg.batch_sizes:
        for input_length in cfg.input_lengths:
            output_tokens = (
                required_decode_output_tokens(
                    cfg.samples_per_length,
                    batch_size,
                    input_length,
                    cfg.engine_chunk_size,
                )
                * _OUTPUT_TOKEN_MULTIPLIER
            )
            configs.append(
                BenchmarkConfig(
                    output_dir=f"{cfg.output_dir}/bs={batch_size}_il={input_length}",
                    seed=cfg.seed,
                    session_generator=SyntheticSessionGeneratorConfig(
                        session_graph=SingleRequestSessionGraphGeneratorConfig(),
                        channels=[
                            TextChannelGeneratorConfig(
                                body_length_generator=FixedLengthGeneratorConfig(
                                    value=input_length,
                                ),
                            )
                        ],
                        output_spec=OutputSpecConfig(
                            text=TextOutputSpecConfig(
                                output_length_generator=FixedLengthGeneratorConfig(
                                    value=output_tokens,
                                ),
                            ),
                        ),
                    ),
                    traffic_scheduler=SequentialLaunchTrafficConfig(
                        cancel_session_on_failure=False,
                    ),
                    evaluators=[
                        PerformanceEvaluatorConfig(
                            stream_metrics=False,
                            text_channel=TextChannelPerformanceConfig(
                                decode_window_enabled=True,
                                decode_window_config=DecodeWindowConfig(
                                    min_active_requests="max_observed",
                                    selection_strategy="all",
                                ),
                            ),
                        ),
                    ],
                    client=_build_client_config(cfg),
                    runtime=RuntimeConfig(
                        max_sessions=batch_size,
                        num_client_threads=batch_size,
                        benchmark_timeout=cfg.benchmark_timeout,
                        pregenerate_sessions=True,
                    ),
                    trace_recorder=TraceRecorderConfig(enabled=False),
                )
            )
    return configs
