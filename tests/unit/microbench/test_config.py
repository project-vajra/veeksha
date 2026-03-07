"""Tests for microbenchmark config dataclass."""

import pytest

from veeksha.microbench.config import (
    BaseMicrobenchmarkConfig,
    DecodeMicrobenchmarkConfig,
    PrefillMicrobenchmarkConfig,
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


class TestInheritance:
    def test_prefill_inherits_base(self):
        assert issubclass(PrefillMicrobenchmarkConfig, BaseMicrobenchmarkConfig)

    def test_decode_inherits_base(self):
        assert issubclass(DecodeMicrobenchmarkConfig, BaseMicrobenchmarkConfig)


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
