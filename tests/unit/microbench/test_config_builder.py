"""Tests for microbenchmark config builder."""

from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.config.generator.length import FixedLengthGeneratorConfig, StairLengthGeneratorConfig
from veeksha.config.generator.session_graph import SingleRequestSessionGraphGeneratorConfig
from veeksha.config.traffic import ConcurrentTrafficConfig, SequentialLaunchTrafficConfig
from veeksha.microbench.config import DecodeMicrobenchmarkConfig, PrefillMicrobenchmarkConfig
from veeksha.microbench.decode import required_decode_output_tokens
from veeksha.microbench.decode import build_benchmark_configs as build_decode_configs
from veeksha.microbench.prefill import build_benchmark_configs as build_prefill_configs


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
