"""Tests for microbenchmark config builder."""

from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.config.generator.length import FixedLengthGeneratorConfig, StairLengthGeneratorConfig
from veeksha.config.generator.session_graph import SingleRequestSessionGraphGeneratorConfig
from veeksha.config.traffic import ConcurrentTrafficConfig, RateTrafficConfig, SequentialLaunchTrafficConfig
from veeksha.microbench.config import (
    DecodeMicrobenchmarkConfig,
    ManualStressModeConfig,
    PrefillMicrobenchmarkConfig,
    RangeStressModeConfig,
    StressMicrobenchmarkConfig,
    StressTrafficMode,
)
from veeksha.microbench.decode import required_decode_output_tokens
from veeksha.microbench.decode import build_benchmark_configs as build_decode_configs
from veeksha.microbench.prefill import build_benchmark_configs as build_prefill_configs
from veeksha.microbench.stress import (
    _extract_stress_point,
    _log_spaced_levels,
    build_benchmark_configs as build_stress_configs,
    estimate_max_sessions,
    resolve_levels,
)


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


class TestPrefillExpansion:
    def test_produces_single_config(self):
        cfg = PrefillMicrobenchmarkConfig(input_lengths=[128, 256], samples_per_length=5)
        result = build_prefill_configs(cfg)
        assert len(result) == 1

    def test_stair_generator(self):
        cfg = PrefillMicrobenchmarkConfig(input_lengths=[128, 256, 512], samples_per_length=3)
        bc = build_prefill_configs(cfg)[0]
        body_gen = bc.session_generator.channels[0].body_length_generator
        assert isinstance(body_gen, StairLengthGeneratorConfig)
        assert body_gen.values == [128, 256, 512]
        assert body_gen.repeat_each == 3
        assert body_gen.wrap is False

    def test_max_sessions(self):
        cfg = PrefillMicrobenchmarkConfig(input_lengths=[128, 256], samples_per_length=10)
        bc = build_prefill_configs(cfg)[0]
        assert bc.runtime.max_sessions == 20

    def test_concurrent_1(self):
        bc = build_prefill_configs(PrefillMicrobenchmarkConfig())[0]
        assert isinstance(bc.traffic_scheduler, ConcurrentTrafficConfig)
        assert bc.traffic_scheduler.target_concurrent_sessions == 1

    def test_pregenerate_sessions(self):
        bc = build_prefill_configs(PrefillMicrobenchmarkConfig())[0]
        assert bc.runtime.pregenerate_sessions is True

    def test_single_request_session_graph(self):
        bc = build_prefill_configs(PrefillMicrobenchmarkConfig())[0]
        assert isinstance(bc.session_generator.session_graph, SingleRequestSessionGraphGeneratorConfig)

    def test_output_tokens(self):
        cfg = PrefillMicrobenchmarkConfig(output_tokens=3)
        bc = build_prefill_configs(cfg)[0]
        out_gen = bc.session_generator.output_spec.text.output_length_generator
        assert isinstance(out_gen, FixedLengthGeneratorConfig)
        assert out_gen.value == 3

    def test_output_dir(self):
        cfg = PrefillMicrobenchmarkConfig(output_dir="my_output")
        bc = build_prefill_configs(cfg)[0]
        assert bc.output_dir == "my_output"

    def test_trace_recorder_disabled(self):
        bc = build_prefill_configs(PrefillMicrobenchmarkConfig())[0]
        assert bc.trace_recorder.enabled is False

    def test_stream_metrics_disabled(self):
        bc = build_prefill_configs(PrefillMicrobenchmarkConfig())[0]
        assert isinstance(bc.evaluators[0], PerformanceEvaluatorConfig)
        assert bc.evaluators[0].stream_metrics is False

    def test_client_fields(self):
        cfg = PrefillMicrobenchmarkConfig(model="my-model", api_base="http://x", api_key="k", max_tokens_param="mt")
        bc = build_prefill_configs(cfg)[0]
        assert bc.client.model == "my-model"
        assert bc.client.api_base == "http://x"
        assert bc.client.api_key == "k"
        assert bc.client.max_tokens_param == "mt"


# ---------------------------------------------------------------------------
# Decode output_tokens formula
# ---------------------------------------------------------------------------


class TestDecodeOutputTokens:
    def test_single_batch(self):
        # batch_size=1 → no ramp-up
        assert required_decode_output_tokens(100, batch_size=1, input_length=1024, chunk_size=512) == 100

    def test_ramp_up(self):
        # batch_size=4, input_length=1024, chunk_size=512
        # effective_chunk = 512 - 4 = 508
        # iters_per_prefill = ceil(1024 / 508) = 3
        # ramp_up = 3 * 3 = 9
        assert required_decode_output_tokens(100, batch_size=4, input_length=1024, chunk_size=512) == 109

    def test_exact_divisible(self):
        # input_length=500, chunk_size=510, batch_size=10
        # effective = 500, iters = ceil(500/500) = 1
        # ramp_up = 9 * 1 = 9
        assert required_decode_output_tokens(50, batch_size=10, input_length=500, chunk_size=510) == 59


# ---------------------------------------------------------------------------
# Decode expansion
# ---------------------------------------------------------------------------


class TestDecodeExpansion:
    def test_cartesian_product_count(self):
        cfg = DecodeMicrobenchmarkConfig(batch_sizes=[2, 4], input_lengths=[128, 256, 512])
        result = build_decode_configs(cfg)
        assert len(result) == 6

    def test_sequential_launch_scheduler(self):
        cfg = DecodeMicrobenchmarkConfig(batch_sizes=[4, 8], input_lengths=[128])
        result = build_decode_configs(cfg)
        assert isinstance(result[0].traffic_scheduler, SequentialLaunchTrafficConfig)
        assert isinstance(result[1].traffic_scheduler, SequentialLaunchTrafficConfig)

    def test_decode_window_enabled(self):
        cfg = DecodeMicrobenchmarkConfig(batch_sizes=[2], input_lengths=[128])
        bc = build_decode_configs(cfg)[0]
        perf = bc.evaluators[0]
        assert isinstance(perf, PerformanceEvaluatorConfig)
        assert perf.text_channel.decode_window_enabled is True
        assert perf.text_channel.decode_window_config.min_active_requests == "max_observed"
        assert perf.text_channel.decode_window_config.selection_strategy == "all"

    def test_param_named_output_dirs(self):
        cfg = DecodeMicrobenchmarkConfig(batch_sizes=[2, 4], input_lengths=[128, 256], output_dir="out")
        result = build_decode_configs(cfg)
        dirs = {bc.output_dir for bc in result}
        assert dirs == {
            "out/bs=2_il=128", "out/bs=2_il=256",
            "out/bs=4_il=128", "out/bs=4_il=256",
        }

    def test_output_tokens_computed(self):
        # batch_size=4, input_length=1024, chunk_size=512, samples=100 → 109
        cfg = DecodeMicrobenchmarkConfig(batch_sizes=[4], input_lengths=[1024], engine_chunk_size=512, samples_per_length=100)
        bc = build_decode_configs(cfg)[0]
        out_gen = bc.session_generator.output_spec.text.output_length_generator
        assert isinstance(out_gen, FixedLengthGeneratorConfig)
        assert out_gen.value == 218

    def test_runtime_max_sessions_equals_batch_size(self):
        cfg = DecodeMicrobenchmarkConfig(batch_sizes=[8], input_lengths=[128])
        bc = build_decode_configs(cfg)[0]
        assert bc.runtime.max_sessions == 8

    def test_pregenerate_sessions(self):
        cfg = DecodeMicrobenchmarkConfig(batch_sizes=[1], input_lengths=[128])
        bc = build_decode_configs(cfg)[0]
        assert bc.runtime.pregenerate_sessions is True


# ---------------------------------------------------------------------------
# Stress: level resolution
# ---------------------------------------------------------------------------


class TestResolveLevels:
    def test_manual_sorted_deduped(self):
        cfg = StressMicrobenchmarkConfig(mode=ManualStressModeConfig(concurrency_levels=[8, 2, 4, 2]))
        assert resolve_levels(cfg) == [2, 4, 8]

    def test_range_basic(self):
        cfg = StressMicrobenchmarkConfig(mode=RangeStressModeConfig(concurrency_min=1, concurrency_max=16, concurrency_points=5))
        levels = resolve_levels(cfg)
        assert levels[0] == 1
        assert levels[-1] == 16
        assert len(levels) <= 5  # may be fewer after dedup/rounding

    def test_log_spaced_single_point(self):
        levels = _log_spaced_levels(4, 64, 1)
        assert levels == [4]

    def test_log_spaced_endpoints(self):
        levels = _log_spaced_levels(1, 64, 8)
        assert levels[0] == 1
        assert levels[-1] == 64
        # all positive
        assert all(l >= 1 for l in levels)


class TestEstimateMaxSessions:
    def test_concurrency_mode(self):
        result = estimate_max_sessions(
            level=10, duration=60, output_length=100, max_tps=500.0,
            traffic_mode=StressTrafficMode.FIXED_CLIENTS,
        )
        # estimated = 10 * 60 * 500 / 100 = 3000, * 2 = 6000
        assert result == 6000

    def test_fixed_rate_mode(self):
        result = estimate_max_sessions(
            level=10, duration=60, output_length=100, max_tps=500.0,
            traffic_mode=StressTrafficMode.FIXED_RATE,
        )
        # estimated = 10 * 60 * 2 = 1200, * 2 = 2400
        assert result == 2400

    def test_floor(self):
        # Very low estimate should floor to level * 10
        result = estimate_max_sessions(
            level=4, duration=1, output_length=10000, max_tps=1.0,
            traffic_mode=StressTrafficMode.FIXED_CLIENTS,
        )
        assert result >= 40  # at least level * 10


# ---------------------------------------------------------------------------
# Stress: config expansion
# ---------------------------------------------------------------------------


class TestStressExpansion:
    def test_config_count_matches_levels(self):
        cfg = StressMicrobenchmarkConfig(mode=ManualStressModeConfig(concurrency_levels=[1, 4, 16]))
        result = build_stress_configs(cfg)
        assert len(result) == 3

    def test_concurrency_traffic(self):
        cfg = StressMicrobenchmarkConfig(mode=ManualStressModeConfig(concurrency_levels=[8]))
        bc = build_stress_configs(cfg)[0]
        assert isinstance(bc.traffic_scheduler, ConcurrentTrafficConfig)
        assert bc.traffic_scheduler.target_concurrent_sessions == 8

    def test_fixed_rate_traffic(self):
        cfg = StressMicrobenchmarkConfig(
            mode=ManualStressModeConfig(concurrency_levels=[10]),
            traffic_mode=StressTrafficMode.FIXED_RATE,
        )
        bc = build_stress_configs(cfg)[0]
        assert isinstance(bc.traffic_scheduler, RateTrafficConfig)
        assert bc.traffic_scheduler.interval_generator.arrival_rate == 10.0

    def test_output_dirs_parameterized(self):
        cfg = StressMicrobenchmarkConfig(
            mode=ManualStressModeConfig(concurrency_levels=[1, 4, 16]),
            output_dir="out",
        )
        result = build_stress_configs(cfg)
        dirs = {bc.output_dir for bc in result}
        assert dirs == {"out/c=1", "out/c=4", "out/c=16"}

    def test_fixed_length_generators(self):
        cfg = StressMicrobenchmarkConfig(
            input_length=256,
            output_length=64,
            mode=ManualStressModeConfig(concurrency_levels=[1]),
        )
        bc = build_stress_configs(cfg)[0]
        body = bc.session_generator.channels[0].body_length_generator
        assert isinstance(body, FixedLengthGeneratorConfig)
        assert body.value == 256
        out = bc.session_generator.output_spec.text.output_length_generator
        assert isinstance(out, FixedLengthGeneratorConfig)
        assert out.value == 64

    def test_num_client_threads_ge_concurrency(self):
        cfg = StressMicrobenchmarkConfig(mode=ManualStressModeConfig(concurrency_levels=[32]))
        bc = build_stress_configs(cfg)[0]
        assert bc.runtime.num_client_threads >= 32

    def test_pregenerate_sessions(self):
        cfg = StressMicrobenchmarkConfig(mode=ManualStressModeConfig(concurrency_levels=[1]))
        bc = build_stress_configs(cfg)[0]
        assert bc.runtime.pregenerate_sessions is True

    def test_trace_recorder_disabled(self):
        cfg = StressMicrobenchmarkConfig(mode=ManualStressModeConfig(concurrency_levels=[1]))
        bc = build_stress_configs(cfg)[0]
        assert bc.trace_recorder.enabled is False


# ---------------------------------------------------------------------------
# Stress: metric extraction
# ---------------------------------------------------------------------------


def _make_metrics(n: int, dispatch_start: float = 100.0) -> list[dict]:
    """Create n synthetic request-level metrics records."""
    records = []
    for i in range(n):
        t = dispatch_start + i * 0.5
        records.append({
            "scheduler_dispatched_at": t,
            "client_picked_up_at": t + 0.01,
            "client_completed_at": t + 0.3,
            "num_output_tokens": 64,
            "end_to_end_latency": 0.29,
            "ttfc": 0.05 + i * 0.001,
            "tpot": 0.004 + i * 0.0001,
            "session_id": i,
            "target_num_delta_prompt_tokens": 512,
        })
    return records


class TestExtractStressPoint:
    def test_basic_extraction(self):
        metrics = _make_metrics(20, dispatch_start=100.0)
        point = _extract_stress_point(4, metrics, warmup_seconds=2, input_length=512)
        assert point is not None
        assert point.level == 4
        assert point.output_throughput > 0
        assert point.input_throughput > 0
        assert point.num_requests > 0
        assert point.num_requests < 20  # some filtered by warmup

    def test_warmup_filters_early_requests(self):
        metrics = _make_metrics(10, dispatch_start=0.0)
        # warmup=3s, requests complete at 0.3, 0.8, 1.3, ... so several filtered
        point = _extract_stress_point(1, metrics, warmup_seconds=3, input_length=512)
        assert point is not None
        assert point.num_requests < 10

    def test_too_few_requests_returns_none(self):
        metrics = _make_metrics(1)
        point = _extract_stress_point(1, metrics, warmup_seconds=0, input_length=512)
        assert point is None

    def test_empty_metrics_returns_none(self):
        point = _extract_stress_point(1, [], warmup_seconds=0, input_length=512)
        assert point is None

    def test_interactivity_computed(self):
        metrics = _make_metrics(20, dispatch_start=0.0)
        point = _extract_stress_point(4, metrics, warmup_seconds=0, input_length=512)
        assert point is not None
        # Interactivity = 1 / tpot
        assert point.interactivity_p50 > 0
        assert point.interactivity_p99 > 0
        # P50 interactivity should be >= P99 (better perf at median)
        assert point.interactivity_p50 >= point.interactivity_p99
