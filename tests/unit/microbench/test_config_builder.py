"""Tests for microbenchmark config builder."""

import math

import pytest

from veeksha.config.evaluator import DecodeWindowConfig, PerformanceEvaluatorConfig
from veeksha.config.generator.length import FixedLengthGeneratorConfig, StairLengthGeneratorConfig
from veeksha.config.generator.session_graph import SingleRequestSessionGraphGeneratorConfig
from veeksha.config.traffic import ConcurrentTrafficConfig, SequentialLaunchTrafficConfig
from veeksha.microbench.config import MicrobenchmarkConfig
from veeksha.microbench.config_builder import required_decode_output_tokens, required_mixed_output_tokens, build_benchmark_configs


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


class TestPrefillExpansion:
    def test_produces_single_config(self):
        cfg = MicrobenchmarkConfig(type="prefill",input_lengths=[128, 256], samples_per_length=5)
        result = build_benchmark_configs(cfg)
        assert len(result) == 1

    def test_stair_generator(self):
        cfg = MicrobenchmarkConfig(type="prefill",input_lengths=[128, 256, 512], samples_per_length=3)
        bc = build_benchmark_configs(cfg)[0]
        body_gen = bc.session_generator.channels[0].body_length_generator
        assert isinstance(body_gen, StairLengthGeneratorConfig)
        assert body_gen.values == [128, 256, 512]
        assert body_gen.repeat_each == 3
        assert body_gen.wrap is False

    def test_max_sessions(self):
        cfg = MicrobenchmarkConfig(type="prefill",input_lengths=[128, 256], samples_per_length=10)
        bc = build_benchmark_configs(cfg)[0]
        assert bc.runtime.max_sessions == 20

    def test_concurrent_1(self):
        bc = build_benchmark_configs(MicrobenchmarkConfig(type="prefill"))[0]
        assert isinstance(bc.traffic_scheduler, ConcurrentTrafficConfig)
        assert bc.traffic_scheduler.target_concurrent_sessions == 1

    def test_pregenerate_sessions(self):
        bc = build_benchmark_configs(MicrobenchmarkConfig(type="prefill"))[0]
        assert bc.runtime.pregenerate_sessions is True

    def test_single_request_session_graph(self):
        bc = build_benchmark_configs(MicrobenchmarkConfig(type="prefill"))[0]
        assert isinstance(bc.session_generator.session_graph, SingleRequestSessionGraphGeneratorConfig)

    def test_output_tokens(self):
        cfg = MicrobenchmarkConfig(type="prefill",output_tokens=3)
        bc = build_benchmark_configs(cfg)[0]
        out_gen = bc.session_generator.output_spec.text.output_length_generator
        assert isinstance(out_gen, FixedLengthGeneratorConfig)
        assert out_gen.value == 3

    def test_output_dir(self):
        cfg = MicrobenchmarkConfig(type="prefill",output_dir="my_output")
        bc = build_benchmark_configs(cfg)[0]
        assert bc.output_dir == "my_output"

    def test_trace_recorder_disabled(self):
        bc = build_benchmark_configs(MicrobenchmarkConfig(type="prefill"))[0]
        assert bc.trace_recorder.enabled is False

    def test_stream_metrics_disabled(self):
        bc = build_benchmark_configs(MicrobenchmarkConfig(type="prefill"))[0]
        assert isinstance(bc.evaluators[0], PerformanceEvaluatorConfig)
        assert bc.evaluators[0].stream_metrics is False

    def test_client_fields(self):
        cfg = MicrobenchmarkConfig(type="prefill",model="my-model", api_base="http://x", api_key="k", max_tokens_param="mt")
        bc = build_benchmark_configs(cfg)[0]
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
        cfg = MicrobenchmarkConfig(type="decode",batch_sizes=[2, 4], input_lengths=[128, 256, 512])
        result = build_benchmark_configs(cfg)
        assert len(result) == 6

    def test_sequential_launch_scheduler(self):
        cfg = MicrobenchmarkConfig(type="decode",batch_sizes=[4, 8], input_lengths=[128])
        result = build_benchmark_configs(cfg)
        assert isinstance(result[0].traffic_scheduler, SequentialLaunchTrafficConfig)
        assert isinstance(result[1].traffic_scheduler, SequentialLaunchTrafficConfig)

    def test_decode_window_enabled(self):
        cfg = MicrobenchmarkConfig(type="decode",batch_sizes=[2], input_lengths=[128])
        bc = build_benchmark_configs(cfg)[0]
        perf = bc.evaluators[0]
        assert isinstance(perf, PerformanceEvaluatorConfig)
        assert perf.text_channel.decode_window_enabled is True
        assert perf.text_channel.decode_window_config.min_active_requests == "max_observed"
        assert perf.text_channel.decode_window_config.selection_strategy == "all"

    def test_param_named_output_dirs(self):
        cfg = MicrobenchmarkConfig(type="decode",batch_sizes=[2, 4], input_lengths=[128, 256], output_dir="out")
        result = build_benchmark_configs(cfg)
        dirs = {bc.output_dir for bc in result}
        assert dirs == {
            "out/bs=2_il=128", "out/bs=2_il=256",
            "out/bs=4_il=128", "out/bs=4_il=256",
        }

    def test_output_tokens_computed(self):
        # batch_size=4, input_length=1024, chunk_size=512, samples=100 → 109
        cfg = MicrobenchmarkConfig(type="decode",batch_sizes=[4], input_lengths=[1024], engine_chunk_size=512, samples_per_length=100)
        bc = build_benchmark_configs(cfg)[0]
        out_gen = bc.session_generator.output_spec.text.output_length_generator
        assert isinstance(out_gen, FixedLengthGeneratorConfig)
        assert out_gen.value == 218

    def test_runtime_max_sessions_equals_batch_size(self):
        cfg = MicrobenchmarkConfig(type="decode",batch_sizes=[8], input_lengths=[128])
        bc = build_benchmark_configs(cfg)[0]
        assert bc.runtime.max_sessions == 8

    def test_pregenerate_sessions(self):
        cfg = MicrobenchmarkConfig(type="decode",batch_sizes=[1], input_lengths=[128])
        bc = build_benchmark_configs(cfg)[0]
        assert bc.runtime.pregenerate_sessions is True


# ---------------------------------------------------------------------------
# Mixed batch output_tokens formula
# ---------------------------------------------------------------------------


class TestMixedOutputTokens:
    def test_no_ramp_up_single_batch(self):
        # batch_size=1 → ramp_up=0, only interference + samples
        result = required_mixed_output_tokens(
            samples_per_length=50,
            batch_size=1,
            decode_input_length=1024,
            chunk_size=512,
            num_prefill_requests=5,
            incremental_prefill_size=256,
        )
        # interference = 5 * ceil(256 / (512 - 1)) = 5 * 1 = 5
        assert result == 50 + 5

    def test_with_ramp_up(self):
        result = required_mixed_output_tokens(
            samples_per_length=100,
            batch_size=4,
            decode_input_length=1024,
            chunk_size=512,
            num_prefill_requests=10,
            incremental_prefill_size=256,
        )
        # ramp_up = 3 * ceil(1024 / 508) = 3 * 3 = 9  (actually ceil(1024/508)=3)
        # interference = 10 * ceil(256 / 508) = 10 * 1 = 10
        assert result == 100 + 9 + 10


# ---------------------------------------------------------------------------
# Mixed batch expansion
# ---------------------------------------------------------------------------


class TestMixedBatchExpansion:
    def test_cartesian_product_count(self):
        """2 configs per (batch_size, decode_input_length): warmup + benchmark."""
        cfg = MicrobenchmarkConfig(type="mixed",batch_sizes=[2, 4], decode_input_lengths=[512, 1024])
        result = build_benchmark_configs(cfg)
        assert len(result) == 8  # 2 batch_sizes * 2 input_lengths * 2 (warmup+bench)

    def test_warmup_config_structure(self):
        """First config of each pair is a warmup: 1 session, prefill_kv_length tokens."""
        cfg = MicrobenchmarkConfig(type="mixed",
            batch_sizes=[4],
            decode_input_lengths=[512],
            prefill_kv_lengths=[512],
            engine_chunk_size=512,
            samples_per_length=10,
        )
        warmup, bench = build_benchmark_configs(cfg)
        assert warmup.output_dir.endswith("/warmup")
        assert warmup.runtime.max_sessions == 1
        assert warmup.traffic_scheduler.target_concurrent_sessions == 1
        body_gen = warmup.session_generator.channels[0].body_length_generator
        assert isinstance(body_gen, FixedLengthGeneratorConfig)
        assert body_gen.value == 512

    def test_warmup_shared_prefix_ratio(self):
        """Warmup uses shared_prefix_ratio=1.0 for prefix cache population."""
        cfg = MicrobenchmarkConfig(type="mixed",batch_sizes=[2], decode_input_lengths=[512])
        warmup = build_benchmark_configs(cfg)[0]
        assert warmup.session_generator.channels[0].shared_prefix_ratio == 1.0

    def test_benchmark_total_sessions(self):
        """Benchmark sessions = batch_size + num_prefill_requests."""
        cfg = MicrobenchmarkConfig(type="mixed",
            batch_sizes=[4],
            decode_input_lengths=[512],
            incremental_prefill_sizes=[256],
            engine_chunk_size=512,
            samples_per_length=10,
        )
        _, bench = build_benchmark_configs(cfg)
        # samples_per_prefill = ceil(256 / (512 - 4)) = ceil(256/508) = 1
        # num_prefill_requests = ceil(10 / 1) = 10
        # total = 4 + 10 = 14
        assert bench.runtime.max_sessions == 14
        assert isinstance(bench.traffic_scheduler, SequentialLaunchTrafficConfig)
        assert bench.runtime.num_client_threads == 14

    def test_benchmark_stair_body_lengths(self):
        """Body stair: [decode_len]*bs + [kv+delta]*num_prefill."""
        cfg = MicrobenchmarkConfig(type="mixed",
            batch_sizes=[2],
            decode_input_lengths=[1024],
            prefill_kv_lengths=[512],
            incremental_prefill_sizes=[256],
            engine_chunk_size=512,
            samples_per_length=5,
        )
        _, bench = build_benchmark_configs(cfg)
        body_gen = bench.session_generator.channels[0].body_length_generator
        assert isinstance(body_gen, StairLengthGeneratorConfig)
        # decode body = 1024, interference body = 512 + 256 = 768
        # samples_per_prefill = ceil(256 / (512-2)) = ceil(256/510) = 1
        # num_prefill = ceil(5/1) = 5
        assert body_gen.values == [1024] * 2 + [768] * 5

    def test_benchmark_stair_output_lengths(self):
        """Output stair: [decode_out]*bs + [1]*num_prefill."""
        cfg = MicrobenchmarkConfig(type="mixed",
            batch_sizes=[2],
            decode_input_lengths=[1024],
            prefill_kv_lengths=[512],
            incremental_prefill_sizes=[256],
            engine_chunk_size=512,
            samples_per_length=5,
        )
        _, bench = build_benchmark_configs(cfg)
        out_gen = bench.session_generator.output_spec.text.output_length_generator
        assert isinstance(out_gen, StairLengthGeneratorConfig)
        # ramp_up = 1 * ceil(1024/510) = 1 * 3 = 3  (actually ceil(1024/510)=3)
        # interference = 5 * ceil(256/510) = 5 * 1 = 5
        # decode_out = (5 + 3 + 5) * 2 = 26
        expected_out = [26] * 2 + [1] * 5
        assert out_gen.values == expected_out

    def test_decode_window_min_active_is_batch_size(self):
        cfg = MicrobenchmarkConfig(type="mixed",batch_sizes=[8], decode_input_lengths=[512])
        _, bench = build_benchmark_configs(cfg)
        perf = bench.evaluators[0]
        assert isinstance(perf, PerformanceEvaluatorConfig)
        dw = perf.text_channel.decode_window_config
        assert isinstance(dw, DecodeWindowConfig)
        assert dw.min_active_requests == 8
        assert dw.selection_strategy == "all"

    def test_output_dirs(self):
        cfg = MicrobenchmarkConfig(type="mixed",batch_sizes=[2, 4], decode_input_lengths=[512], output_dir="out")
        result = build_benchmark_configs(cfg)
        warmup_dirs = {bc.output_dir for bc in result if "warmup" in bc.output_dir}
        bench_dirs = {bc.output_dir for bc in result if "warmup" not in bc.output_dir}
        assert warmup_dirs == {"out/bs=2_dil=512_kv=512_dp=256/warmup", "out/bs=4_dil=512_kv=512_dp=256/warmup"}
        assert bench_dirs == {"out/bs=2_dil=512_kv=512_dp=256/bench", "out/bs=4_dil=512_kv=512_dp=256/bench"}

    def test_benchmark_shared_prefix_ratio(self):
        """Benchmark config also uses shared_prefix_ratio=1.0."""
        cfg = MicrobenchmarkConfig(type="mixed",batch_sizes=[1], decode_input_lengths=[512])
        _, bench = build_benchmark_configs(cfg)
        assert bench.session_generator.channels[0].shared_prefix_ratio == 1.0
