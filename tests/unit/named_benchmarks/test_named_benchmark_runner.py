from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import StreamingTTSClientConfig, STTClientConfig
from veeksha.config.evaluator import (
    AudioChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.config.named_benchmark import NamedBenchmarkConfig
from veeksha.named_benchmarks.runner import (
    _validate_huggingface_asr_trace,
    _validate_interaction_contract,
    run_named_benchmark,
)
from veeksha.named_benchmarks.schema import Benchmark, DatasetCase

_SOURCE_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialized_hf_dataset(
    tmp_path: Path,
    *,
    expected_rows: int = 2,
    catalog_checksum: str | None = None,
) -> tuple[DatasetCase, Path, Path]:
    dataset_dir = tmp_path / "materialized" / "kathbath"
    dataset_dir.mkdir(parents=True)
    manifest_path = dataset_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(
                {
                    "session_id": row_index,
                    "audio_file": f"audio/{row_index}.wav",
                    "expected_transcript": f"transcript {row_index}",
                    "source_revision": _SOURCE_REVISION,
                    "source_row_index": row_index,
                },
                sort_keys=True,
            )
            + "\n"
            for row_index in range(expected_rows)
        ),
        encoding="utf-8",
    )
    preparation_path = dataset_dir / "preparation.json"
    preparation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "canonical": True,
                "source": {
                    "repo_id": "owner/indic-asr-eval",
                    "revision": _SOURCE_REVISION,
                    "config": "kathbath",
                    "split": "test",
                },
                "selection": {
                    "order": "source_order",
                    "total_rows": expected_rows,
                    "materialized_rows": expected_rows,
                    "max_samples": None,
                },
                "manifest": {
                    "path": "manifest.jsonl",
                    "sha256": _sha256(manifest_path),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    source = {
        "kind": "huggingface",
        "uri": "owner/indic-asr-eval",
        "revision": _SOURCE_REVISION,
        "config": "kathbath",
        "split": "test",
        "expected_rows": expected_rows,
    }
    if catalog_checksum is not None:
        source["checksum"] = catalog_checksum
    dataset = DatasetCase.from_mapping(
        {
            "id": "kathbath",
            "name": "Kathbath",
            "source": source,
            "session_generator": {
                "type": "trace",
                "trace_file": str(manifest_path),
                "flavor": {"type": "audio", "audio_dir": ""},
            },
        },
        index=0,
    )
    return dataset, manifest_path, preparation_path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _asr_language_benchmark() -> Benchmark:
    return Benchmark.from_mapping(
        {
            "schema_version": 1,
            "id": "asr.indic.language-contract.v1",
            "name": "ASR language contract",
            "description": "Validates target language routing coverage.",
            "modality": "asr",
            "input_mode": "streaming",
            "output_mode": "streaming",
            "interaction": {
                "audio_encoding": "pcm_s16le",
                "required_languages": ["bn", "hi", "ta"],
            },
            "client_overrides": {},
            "datasets": [
                {
                    "id": "fixture",
                    "name": "Fixture",
                    "source": {
                        "kind": "fixture",
                        "uri": "fixture://audio",
                        "revision": "fixture-v1",
                        "split": "test",
                    },
                    "session_generator": {
                        "type": "trace",
                        "trace_file": "fixture.jsonl",
                        "flavor": {"type": "audio", "audio_dir": ""},
                    },
                }
            ],
            "execution": {"seed": 42},
            "metrics": [
                {
                    "id": "wer",
                    "role": "primary",
                    "unit": "percent",
                    "dataset_aggregation": {
                        "method": "ratio_of_sums",
                        "numerator": "errors",
                        "denominator": "words",
                        "scale": 100,
                    },
                    "benchmark_aggregation": {
                        "method": "ratio_of_sums",
                        "numerator": "errors",
                        "denominator": "words",
                        "scale": 100,
                    },
                }
            ],
        }
    )


def test_dry_run_compiles_one_ordinary_run_per_dataset_and_target(
    tmp_path: Path,
) -> None:
    target_config = tmp_path / "target.yml"
    target_config.write_text(
        """\
client:
  type: streaming_tts
  provider: vajra
  api_base: http://localhost:8000
  model: fixture-model
  sample_rate: 22050
traffic_scheduler:
  type: concurrent
  target_concurrent_sessions: !expand [1, 2]
  rampup_seconds: 0
""",
        encoding="utf-8",
    )

    benchmark_manifest = tmp_path / "benchmark.yml"
    benchmark_manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "tts.indic.compile-fixture.v1",
                "name": "Compile fixture",
                "description": "Checks target-independent dry-run compilation.",
                "modality": "tts",
                "input_mode": "static",
                "output_mode": "streaming",
                "interaction": {
                    "transport": "websocket",
                    "output_audio_encoding": "pcm_s16le",
                    "output_sample_rate_hz": 24000,
                    "output_channels": 1,
                    "playable_frame_ms": 20,
                },
                "client_overrides": {
                    "sample_rate": 24000,
                    "strict_audio_contract": True,
                },
                "datasets": [
                    {
                        "id": dataset_id,
                        "name": dataset_id,
                        "source": {
                            "kind": "fixture",
                            "uri": f"fixture://{dataset_id}",
                            "revision": "0123456789abcdef",
                            "split": "test",
                        },
                        "client_overrides": {"speed": speed},
                        "session_generator": {
                            "type": "trace",
                            "trace_file": (
                                f"${{DATASET_ROOT}}/{dataset_id}/manifest.jsonl"
                            ),
                            "wrap_mode": False,
                            "flavor": {
                                "type": "seed_tts_text",
                                "dataset_name": "fixture/prompts",
                                "split": "test",
                                "preserve_text": True,
                            },
                        },
                    }
                    for dataset_id, speed in (
                        ("dataset_a", 0.9),
                        ("dataset_b", 1.1),
                    )
                ],
                "execution": {
                    "traffic_scheduler": {
                        "type": "concurrent",
                        "target_concurrent_sessions": 1,
                        "rampup_seconds": 0,
                    },
                    "evaluators": [
                        {
                            "type": "performance",
                            "target_channels": ["audio"],
                            "audio_channel": {
                                "interactivity_enabled": True,
                                "fluidity_frame_ms": 20,
                            },
                            "slos": [],
                        }
                    ],
                    "runtime": {"max_sessions": 1, "benchmark_timeout": 30},
                },
                "metrics": [
                    {
                        "id": "ttfa_ms",
                        "role": "primary",
                        "unit": "ms",
                        "dataset_aggregation": {
                            "method": "distribution",
                            "source": "trigger_to_first_playable_audio_ms",
                            "quantiles": [0.5, 0.9],
                        },
                        "benchmark_aggregation": {
                            "method": "pooled_distribution",
                            "source": "trigger_to_first_playable_audio_ms",
                            "quantiles": [0.5, 0.9],
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    dataset_root = tmp_path / "datasets"
    config = NamedBenchmarkConfig(
        benchmark=str(benchmark_manifest),
        target_config=str(target_config),
        output_dir=str(tmp_path / "results"),
        dataset_root=str(dataset_root),
        dry_run=True,
    )

    parent_dir = run_named_benchmark(config)

    status = json.loads((parent_dir / "run_status.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (parent_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    assert status == {
        "status": "dry_run",
        "compiled_runs": 2,
        "completed_runs": 0,
        "failed_runs": 0,
    }
    assert len(manifest["targets"]) == 1
    assert manifest["runner_revision"] == manifest["runner_state"]["git_commit"]
    assert isinstance(manifest["runner_state"]["dirty"], bool)
    assert len(manifest["runner_state"]["working_tree_diff_sha256"]) == 64
    assert manifest["runner_state"]["target_config_sha256"] == _sha256(target_config)
    assert {child["dataset_id"] for child in manifest["children"]} == {
        "dataset_a",
        "dataset_b",
    }

    compiled_by_dataset = {}
    for child in manifest["children"]:
        path = parent_dir / child["compiled_config"]
        compiled_by_dataset[child["dataset_id"]] = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )

    for dataset_id, expected_speed in (
        ("dataset_a", 0.9),
        ("dataset_b", 1.1),
    ):
        compiled = compiled_by_dataset[dataset_id]
        assert compiled["client"]["provider"] == "vajra"
        assert compiled["client"]["model"] == "fixture-model"
        assert compiled["client"]["sample_rate"] == 24000
        assert compiled["client"]["speed"] == expected_speed
        assert compiled["session_generator"]["trace_file"] == str(
            dataset_root / dataset_id / "manifest.jsonl"
        )
        assert compiled["output_dir"].startswith(str(parent_dir / "targets"))

    # The target config is only a binding; its workload defaults must not leak
    # into either compiled child.
    assert all(
        compiled["session_generator"]["type"] == "trace"
        for compiled in compiled_by_dataset.values()
    )


@pytest.mark.unit
def test_huggingface_asr_trace_accepts_matching_canonical_provenance(
    tmp_path: Path,
) -> None:
    dataset, _, _ = _materialized_hf_dataset(tmp_path)

    _validate_huggingface_asr_trace(dataset, dataset_root="")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("canonical",), False, "noncanonical"),
        (("selection", "max_samples"), 1, "max_samples=null"),
        (("selection", "order"), "random", "selection.order"),
        (("selection", "total_rows"), 1, "expected_rows=2"),
        (("selection", "materialized_rows"), 1, "expected_rows=2"),
        (("source", "repo_id"), "other/repo", "source.repo_id"),
        (("source", "revision"), "f" * 40, "source.revision"),
        (("source", "config"), "fleurs", "source.config"),
        (("source", "split"), "validation", "source.split"),
    ],
)
def test_huggingface_asr_trace_rejects_noncanonical_or_mismatched_preparation(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: object,
    message: str,
) -> None:
    dataset, _, preparation_path = _materialized_hf_dataset(tmp_path)
    metadata = _read_json(preparation_path)
    target = metadata
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = replacement
    _write_json(preparation_path, metadata)

    with pytest.raises(ValueError, match=message):
        _validate_huggingface_asr_trace(dataset, dataset_root="")


@pytest.mark.unit
def test_huggingface_asr_trace_rejects_manifest_modified_after_preparation(
    tmp_path: Path,
) -> None:
    dataset, manifest_path, _ = _materialized_hf_dataset(tmp_path)
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="modified after preparation"):
        _validate_huggingface_asr_trace(dataset, dataset_root="")


@pytest.mark.unit
def test_huggingface_asr_trace_rejects_catalog_checksum_mismatch(
    tmp_path: Path,
) -> None:
    dataset, _, _ = _materialized_hf_dataset(
        tmp_path,
        catalog_checksum="sha256:" + "0" * 64,
    )

    with pytest.raises(ValueError, match="source.checksum"):
        _validate_huggingface_asr_trace(dataset, dataset_root="")


@pytest.mark.unit
def test_huggingface_asr_trace_rejects_rewritten_preparation_for_wrong_rows(
    tmp_path: Path,
) -> None:
    dataset, manifest_path, preparation_path = _materialized_hf_dataset(tmp_path)
    rows = manifest_path.read_text(encoding="utf-8").splitlines()
    rows[1] = json.dumps(
        {
            **json.loads(rows[1]),
            "source_row_index": 99,
        },
        sort_keys=True,
    )
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    metadata = _read_json(preparation_path)
    metadata["manifest"]["sha256"] = _sha256(manifest_path)
    _write_json(preparation_path, metadata)

    with pytest.raises(ValueError, match="source_row_index=99"):
        _validate_huggingface_asr_trace(dataset, dataset_root="")


@pytest.mark.unit
def test_asr_required_languages_must_be_covered_by_request_metadata_target() -> None:
    benchmark = _asr_language_benchmark()
    complete_target = BenchmarkConfig(
        client=STTClientConfig(
            provider="deepgram_nova",
            model="nova-3",
            api_base="https://api.deepgram.com",
            ws_realtime_pacing=True,
            language_mode="request_metadata",
            supported_languages=["ta", "bn", "hi", "en"],
        )
    )
    _validate_interaction_contract(benchmark, complete_target, "fixture")

    incomplete_target = BenchmarkConfig(
        client=STTClientConfig(
            provider="deepgram_nova",
            model="nova-3",
            api_base="https://api.deepgram.com",
            ws_realtime_pacing=True,
            language_mode="request_metadata",
            supported_languages=["hi"],
        )
    )
    with pytest.raises(
        ValueError,
        match=("client.supported_languages does not cover required languages: bn, ta"),
    ):
        _validate_interaction_contract(benchmark, incomplete_target, "fixture")

    fixed_language_target = BenchmarkConfig(
        client=STTClientConfig(
            provider="deepgram_nova",
            model="nova-3",
            api_base="https://api.deepgram.com",
            ws_realtime_pacing=True,
            language_mode="fixed",
            language="hi",
            supported_languages=["bn", "hi", "ta"],
        )
    )
    with pytest.raises(
        ValueError,
        match="language_mode=request_metadata or auto",
    ):
        _validate_interaction_contract(
            benchmark,
            fixed_language_target,
            "fixture",
        )


@pytest.mark.unit
def test_tts_multilingual_pcm_contract_requires_streaming_target_capabilities() -> None:
    benchmark = Benchmark.from_mapping(
        {
            "schema_version": 1,
            "id": "tts.indic.language-contract.v1",
            "name": "TTS language contract",
            "description": "Validates streaming TTS target capabilities.",
            "modality": "tts",
            "input_mode": "static",
            "output_mode": "streaming",
            "interaction": {
                "transport": "websocket",
                "output_audio_encoding": "pcm_s16le",
                "output_sample_rate_hz": 24000,
                "output_channels": 1,
                "playable_frame_ms": 20,
                "required_languages": ["bn", "hi", "ta"],
                "language_metadata_key": "language",
            },
            "client_overrides": {
                "sample_rate": 24000,
                "strict_audio_contract": True,
            },
            "datasets": [
                {
                    "id": "fixture",
                    "name": "Fixture",
                    "source": {
                        "kind": "fixture",
                        "uri": "fixture://prompts",
                        "revision": "fixture-v1",
                        "split": "test",
                    },
                    "session_generator": {
                        "type": "trace",
                        "trace_file": "fixture.jsonl",
                        "flavor": {
                            "type": "seed_tts_text",
                            "dataset_name": "fixture/prompts",
                            "split": "test",
                        },
                    },
                }
            ],
            "execution": {
                "evaluators": [
                    {
                        "type": "performance",
                        "target_channels": ["audio"],
                        "audio_channel": {
                            "interactivity_enabled": True,
                            "fluidity_frame_ms": 20,
                        },
                    }
                ]
            },
            "metrics": [
                {
                    "id": "ttfa",
                    "role": "primary",
                    "unit": "ms",
                    "dataset_aggregation": {
                        "method": "distribution",
                        "source": "trigger_to_first_playable_audio_ms",
                        "quantiles": [0.5],
                    },
                    "benchmark_aggregation": {
                        "method": "pooled_distribution",
                        "source": "trigger_to_first_playable_audio_ms",
                        "quantiles": [0.5],
                    },
                }
            ],
        }
    )

    evaluators = [
        PerformanceEvaluatorConfig(
            target_channels=["audio"],
            audio_channel=AudioChannelPerformanceConfig(fluidity_frame_ms=20),
        )
    ]
    complete_target = BenchmarkConfig(
        client=StreamingTTSClientConfig(
            provider="vajra",
            model="fixture-model",
            api_base="http://localhost:8000",
            sample_rate=24000,
            language_mode="request_metadata",
            supported_languages=["bn", "hi", "ta", "te"],
            strict_audio_contract=True,
        ),
        evaluators=evaluators,
    )
    _validate_interaction_contract(benchmark, complete_target, "fixture")

    wrong_sample_rate = BenchmarkConfig(
        client=StreamingTTSClientConfig(
            provider="vajra",
            model="fixture-model",
            api_base="http://localhost:8000",
            sample_rate=16000,
            language_mode="auto",
            supported_languages=["bn", "hi", "ta"],
            strict_audio_contract=True,
        ),
        evaluators=evaluators,
    )
    with pytest.raises(ValueError, match="benchmark requires 24000 Hz"):
        _validate_interaction_contract(benchmark, wrong_sample_rate, "fixture")

    incomplete_target = BenchmarkConfig(
        client=StreamingTTSClientConfig(
            provider="vajra",
            model="fixture-model",
            api_base="http://localhost:8000",
            sample_rate=24000,
            language_mode="auto",
            supported_languages=["hi"],
            strict_audio_contract=True,
        ),
        evaluators=evaluators,
    )
    with pytest.raises(ValueError, match="required languages: bn, ta"):
        _validate_interaction_contract(benchmark, incomplete_target, "fixture")
