"""Expand simplified microbenchmark configs into full BenchmarkConfig instances."""

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
from veeksha.config.generator.requested_output import OutputSpecConfig, TextOutputSpecConfig
from veeksha.config.generator.session import SyntheticSessionGeneratorConfig
from veeksha.config.generator.session_graph import SingleRequestSessionGraphGeneratorConfig
from veeksha.config.runtime import RuntimeConfig
from veeksha.config.trace_recorder import TraceRecorderConfig
from veeksha.config.traffic import ConcurrentTrafficConfig, SequentialLaunchTrafficConfig
from veeksha.microbench.config import MicrobenchmarkConfig

_OUTPUT_TOKEN_MULTIPLIER = 2


def expand(cfg: MicrobenchmarkConfig) -> list[BenchmarkConfig]:
    """Convert a simplified microbenchmark config into a list of full BenchmarkConfigs."""
    if cfg.type == "prefill":
        return _expand_prefill(cfg)
    elif cfg.type == "decode":
        return _expand_decode(cfg)
    elif cfg.type == "mixed":
        return _expand_mixed(cfg)
    else:
        raise ValueError(f"Unknown config type: {cfg.type}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(cfg: MicrobenchmarkConfig) -> OpenAIChatCompletionsClientConfig:
    return OpenAIChatCompletionsClientConfig(
        model=cfg.model,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        request_timeout=cfg.request_timeout,
        max_tokens_param=cfg.max_tokens_param,
        ignore_eos=cfg.ignore_eos,
    )


def _prefill_iters(input_length: int, chunk_size: int, active_decodes: int) -> int:
    """Iterations needed to prefill one request given active decode slots.

    Each iteration has a token budget of *chunk_size*.  Active decode
    requests consume one token each, leaving the rest for prefill.
    """
    effective_chunk = chunk_size - active_decodes
    assert effective_chunk > 0, (
        f"chunk_size ({chunk_size}) must exceed active_decodes ({active_decodes})"
    )
    return math.ceil(input_length / effective_chunk)


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


def _expand_prefill(cfg: MicrobenchmarkConfig) -> list[BenchmarkConfig]:
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
            client=_make_client(cfg),
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


def _decode_output_tokens(
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
    ramp_up = (batch_size - 1) * _prefill_iters(input_length, chunk_size, batch_size)
    return samples_per_length + ramp_up


def _expand_decode(cfg: MicrobenchmarkConfig) -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    for batch_size in cfg.batch_sizes:
        for input_length in cfg.input_lengths:
            output_tokens = _decode_output_tokens(
                cfg.samples_per_length, batch_size, input_length, cfg.engine_chunk_size,
            ) * _OUTPUT_TOKEN_MULTIPLIER
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
                    client=_make_client(cfg),
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


# ---------------------------------------------------------------------------
# Mixed batch (incremental prefill interference)
# ---------------------------------------------------------------------------


def _mixed_output_tokens(
    samples_per_length: int,
    batch_size: int,
    decode_input_length: int,
    chunk_size: int,
    num_prefill_requests: int,
    incremental_prefill_size: int,
) -> int:
    """Compute output tokens for decode requests in a mixed-batch run.

    Decode request 0 must survive through two phases after it enters
    decode mode, plus extra iterations for measurement:

    1. Decode ramp-up:  (batch_size - 1) * ceil(decode_input_length / (chunk_size - batch_size))
    2. Interference:  num_prefill_requests * ceil(incremental_prefill_size / (chunk_size - batch_size))

    Plus *samples_per_length* pure-decode iterations after interference ends.

    Note: the cache warmup prefill happens *before* decode requests start
    prefilling (FCFS ordering), so it does not consume decode output tokens.
    """
    # Phase 1: decode ramp-up
    if batch_size == 1:
        ramp_up = 0
    else:
        ramp_up = (batch_size - 1) * _prefill_iters(
            decode_input_length, chunk_size, batch_size,
        )

    # Phase 3: interference
    interference = num_prefill_requests * _prefill_iters(
        incremental_prefill_size, chunk_size, batch_size,
    )

    return samples_per_length + ramp_up + interference


def _expand_mixed(cfg: MicrobenchmarkConfig) -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []

    for batch_size in cfg.batch_sizes:
        for decode_input_length in cfg.decode_input_lengths:
            for prefill_kv_length in cfg.prefill_kv_lengths:
                for incremental_prefill_size in cfg.incremental_prefill_sizes:
                    _expand_one_mixed(
                        configs, cfg, batch_size, decode_input_length,
                        prefill_kv_length, incremental_prefill_size,
                    )
    return configs


def _expand_one_mixed(
    configs: list[BenchmarkConfig],
    cfg: MicrobenchmarkConfig,
    batch_size: int,
    decode_input_length: int,
    prefill_kv_length: int,
    incremental_prefill_size: int,
) -> None:
    # How many TBT samples one interference prefill request produces
    samples_per_prefill = _prefill_iters(
        incremental_prefill_size, cfg.engine_chunk_size, batch_size,
    )
    num_prefill_requests = math.ceil(cfg.samples_per_length / samples_per_prefill)

    decode_out = _mixed_output_tokens(
        samples_per_length=cfg.samples_per_length,
        batch_size=batch_size,
        decode_input_length=decode_input_length,
        chunk_size=cfg.engine_chunk_size,
        num_prefill_requests=num_prefill_requests,
        incremental_prefill_size=incremental_prefill_size,
    ) * _OUTPUT_TOKEN_MULTIPLIER

    tag = f"bs={batch_size}_dil={decode_input_length}_kv={prefill_kv_length}_dp={incremental_prefill_size}"

    # -- Phase 0: cache warmup (separate config, runs first) ----------
    configs.append(
        BenchmarkConfig(
            output_dir=f"{cfg.output_dir}/{tag}/warmup",
            seed=cfg.seed,
            session_generator=SyntheticSessionGeneratorConfig(
                session_graph=SingleRequestSessionGraphGeneratorConfig(),
                channels=[
                    TextChannelGeneratorConfig(
                        body_length_generator=FixedLengthGeneratorConfig(
                            value=prefill_kv_length,
                        ),
                        shared_prefix_ratio=1.0,
                    )
                ],
                output_spec=OutputSpecConfig(
                    text=TextOutputSpecConfig(
                        output_length_generator=FixedLengthGeneratorConfig(value=1),
                    ),
                ),
            ),
            traffic_scheduler=ConcurrentTrafficConfig(
                target_concurrent_sessions=1,
                rampup_seconds=0,
                cancel_session_on_failure=False,
            ),
            evaluators=[PerformanceEvaluatorConfig(stream_metrics=False)],
            client=_make_client(cfg),
            runtime=RuntimeConfig(
                max_sessions=1,
                benchmark_timeout=cfg.benchmark_timeout,
                pregenerate_sessions=True,
            ),
            trace_recorder=TraceRecorderConfig(enabled=False),
        )
    )

    # -- Phase 1+2: decode + interference (main benchmark config) ------
    bench_sessions = batch_size + num_prefill_requests

    body_values = (
        [decode_input_length] * batch_size
        + [prefill_kv_length + incremental_prefill_size] * num_prefill_requests
    )
    output_values = (
        [decode_out] * batch_size
        + [1] * num_prefill_requests
    )

    configs.append(
        BenchmarkConfig(
            output_dir=f"{cfg.output_dir}/{tag}/bench",
            seed=cfg.seed,
            session_generator=SyntheticSessionGeneratorConfig(
                session_graph=SingleRequestSessionGraphGeneratorConfig(),
                channels=[
                    TextChannelGeneratorConfig(
                        body_length_generator=StairLengthGeneratorConfig(
                            values=body_values,
                            repeat_each=1,
                            wrap=False,
                        ),
                        shared_prefix_ratio=1.0,
                    )
                ],
                output_spec=OutputSpecConfig(
                    text=TextOutputSpecConfig(
                        output_length_generator=StairLengthGeneratorConfig(
                            values=output_values,
                            repeat_each=1,
                            wrap=False,
                        ),
                    ),
                ),
            ),
            traffic_scheduler=SequentialLaunchTrafficConfig(
                cancel_session_on_failure=False,
                ordering="prefill",
            ),
            evaluators=[
                PerformanceEvaluatorConfig(
                    stream_metrics=False,
                    text_channel=TextChannelPerformanceConfig(
                        decode_window_enabled=True,
                        decode_window_config=DecodeWindowConfig(
                            min_active_requests=batch_size,
                            selection_strategy="all",
                        ),
                    ),
                ),
            ],
            client=_make_client(cfg),
            runtime=RuntimeConfig(
                max_sessions=bench_sessions,
                num_client_threads=bench_sessions,
                benchmark_timeout=cfg.benchmark_timeout,
                pregenerate_sessions=True,
            ),
            trace_recorder=TraceRecorderConfig(enabled=False),
        )
    )
