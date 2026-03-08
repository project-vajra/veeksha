"""Tests for microbenchmark config dataclass."""

import pytest

from veeksha.microbench.config import (
    AutoStressConfig,
    BaseMicrobenchmarkConfig,
    DecodeMicrobenchmarkConfig,
    ManualStressConfig,
    PrefillMicrobenchmarkConfig,
    RangeStressConfig,
    StressMicrobenchmarkConfig,
    StressTrafficMode,
)


class TestPrefillConfig:
    def test_defaults(self):
        cfg = PrefillMicrobenchmarkConfig()
        assert cfg.input_lengths == [128, 256, 512, 1024]
        assert cfg.output_tokens == 1
        assert cfg.samples_per_length == 10

    def test_custom_values(self):
        cfg = PrefillMicrobenchmarkConfig(input_lengths=[64, 128], samples_per_length=5, output_tokens=2)
        assert cfg.input_lengths == [64, 128]
        assert cfg.samples_per_length == 5
        assert cfg.output_tokens == 2

    def test_empty_input_lengths(self):
        with pytest.raises(ValueError, match="input_lengths must be non-empty"):
            PrefillMicrobenchmarkConfig(input_lengths=[])

    def test_non_positive_output_tokens(self):
        with pytest.raises(ValueError, match="output_tokens must be positive"):
            PrefillMicrobenchmarkConfig(output_tokens=0)

    def test_non_positive_samples_per_length(self):
        with pytest.raises(ValueError, match="samples_per_length must be positive"):
            PrefillMicrobenchmarkConfig(samples_per_length=0)

    def test_frozen(self):
        cfg = PrefillMicrobenchmarkConfig()
        with pytest.raises(AttributeError):
            cfg.output_tokens = 5  # type: ignore[misc]


class TestDecodeConfig:
    def test_defaults(self):
        cfg = DecodeMicrobenchmarkConfig()
        assert cfg.input_lengths == [128, 256, 512, 1024]
        assert cfg.batch_sizes == [1, 2, 4, 8]
        assert cfg.engine_chunk_size == 512
        assert cfg.samples_per_length == 10

    def test_empty_input_lengths(self):
        with pytest.raises(ValueError, match="input_lengths must be non-empty"):
            DecodeMicrobenchmarkConfig(input_lengths=[])

    def test_empty_batch_sizes(self):
        with pytest.raises(ValueError, match="batch_sizes must be non-empty"):
            DecodeMicrobenchmarkConfig(batch_sizes=[])

    def test_non_positive_samples_per_length(self):
        with pytest.raises(ValueError, match="samples_per_length must be positive"):
            DecodeMicrobenchmarkConfig(samples_per_length=0)

    def test_non_positive_engine_chunk_size(self):
        with pytest.raises(ValueError, match="engine_chunk_size must be positive"):
            DecodeMicrobenchmarkConfig(engine_chunk_size=0)

    def test_batch_size_ge_chunk_size(self):
        with pytest.raises(ValueError, match="batch_size 512 must be less than engine_chunk_size 512"):
            DecodeMicrobenchmarkConfig(batch_sizes=[512], engine_chunk_size=512)


class TestManualStressConfig:
    def test_defaults(self):
        cfg = ManualStressConfig()
        assert cfg.input_length == 512
        assert cfg.output_length == 128
        assert cfg.point_duration == 120
        assert cfg.warmup_duration == 10
        assert cfg.traffic_mode == StressTrafficMode.FIXED_CLIENTS
        assert cfg.max_tokens_per_second_estimate == 500.0
        assert cfg.concurrency_levels == [1, 2, 4, 8, 16, 32]

    def test_custom_values(self):
        cfg = ManualStressConfig(
            input_length=256, output_length=64, concurrency_levels=[1, 4, 16]
        )
        assert cfg.input_length == 256
        assert cfg.output_length == 64
        assert cfg.concurrency_levels == [1, 4, 16]

    def test_negative_input_length(self):
        with pytest.raises(ValueError, match="input_length must be positive"):
            ManualStressConfig(input_length=0)

    def test_negative_output_length(self):
        with pytest.raises(ValueError, match="output_length must be positive"):
            ManualStressConfig(output_length=-1)

    def test_warmup_exceeds_duration(self):
        with pytest.raises(ValueError, match="point_duration must exceed warmup_duration"):
            ManualStressConfig(point_duration=10, warmup_duration=10)

    def test_empty_concurrency_levels(self):
        with pytest.raises(ValueError, match="concurrency_levels must be non-empty"):
            ManualStressConfig(concurrency_levels=[])

    def test_non_positive_concurrency_level(self):
        with pytest.raises(ValueError, match="all concurrency_levels must be positive"):
            ManualStressConfig(concurrency_levels=[1, 0, 4])

    def test_invalid_traffic_mode(self):
        with pytest.raises(ValueError, match="traffic_mode must be"):
            ManualStressConfig(traffic_mode="invalid")  # type: ignore[arg-type]

    def test_fixed_rate_traffic_mode(self):
        cfg = ManualStressConfig(traffic_mode=StressTrafficMode.FIXED_RATE)
        assert cfg.traffic_mode == StressTrafficMode.FIXED_RATE

    def test_frozen(self):
        cfg = ManualStressConfig()
        with pytest.raises(AttributeError):
            cfg.input_length = 256  # type: ignore[misc]


class TestRangeStressConfig:
    def test_defaults(self):
        cfg = RangeStressConfig()
        assert cfg.concurrency_min == 1
        assert cfg.concurrency_max == 64
        assert cfg.concurrency_points == 8

    def test_min_ge_max(self):
        with pytest.raises(ValueError, match="concurrency_min must be less than concurrency_max"):
            RangeStressConfig(concurrency_min=64, concurrency_max=64)

    def test_non_positive_points(self):
        with pytest.raises(ValueError, match="concurrency_points must be positive"):
            RangeStressConfig(concurrency_points=0)

    def test_inherits_base_validation(self):
        with pytest.raises(ValueError, match="input_length must be positive"):
            RangeStressConfig(input_length=-1)


class TestAutoStressConfig:
    def test_defaults(self):
        cfg = AutoStressConfig()
        assert cfg.auto_throughput_threshold == 0.05
        assert cfg.auto_max_probes == 20
        assert cfg.auto_fill_points == 8

    def test_non_positive_max_probes(self):
        with pytest.raises(ValueError, match="auto_max_probes must be positive"):
            AutoStressConfig(auto_max_probes=0)

    def test_non_positive_fill_points(self):
        with pytest.raises(ValueError, match="auto_fill_points must be positive"):
            AutoStressConfig(auto_fill_points=0)

    def test_inherits_base_validation(self):
        with pytest.raises(ValueError, match="output_length must be positive"):
            AutoStressConfig(output_length=0)


class TestInheritance:
    def test_prefill_inherits_base(self):
        assert issubclass(PrefillMicrobenchmarkConfig, BaseMicrobenchmarkConfig)

    def test_decode_inherits_base(self):
        assert issubclass(DecodeMicrobenchmarkConfig, BaseMicrobenchmarkConfig)

    def test_stress_hierarchy(self):
        assert issubclass(StressMicrobenchmarkConfig, BaseMicrobenchmarkConfig)
        assert issubclass(ManualStressConfig, StressMicrobenchmarkConfig)
        assert issubclass(RangeStressConfig, StressMicrobenchmarkConfig)
        assert issubclass(AutoStressConfig, StressMicrobenchmarkConfig)

    def test_stress_mode_classvar(self):
        assert ManualStressConfig.STRESS_MODE == "manual"
        assert RangeStressConfig.STRESS_MODE == "range"
        assert AutoStressConfig.STRESS_MODE == "auto"


class TestCommonFields:
    def test_shared_defaults(self):
        cfg = PrefillMicrobenchmarkConfig()
        assert cfg.model == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert cfg.api_base == "http://localhost:8000/v1"
        assert cfg.api_key == "dummy"
        assert cfg.output_dir == "microbench_output"
        assert cfg.seed == 42
        assert cfg.request_timeout == 120
        assert cfg.benchmark_timeout == 600
        assert cfg.max_tokens_param == "max_tokens"
        assert cfg.ignore_eos is True
        assert cfg.validate_only is False
        assert cfg.skip_validation is False
