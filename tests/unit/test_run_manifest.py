"""Unit tests for run_manifest start/finalize helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from veeksha.benchmark_utils import (
    RUN_MANIFEST_NAME,
    finalize_run_manifest,
    write_run_manifest_start,
)
from veeksha.config.benchmark import BenchmarkConfig


@pytest.mark.unit
def test_run_manifest_start_and_finalize(tmp_path: Path) -> None:
    cfg = BenchmarkConfig(output_dir=str(tmp_path))
    write_run_manifest_start(cfg, config_sha1="deadbeef", tokenizer_model="m")
    path = tmp_path / RUN_MANIFEST_NAME
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["inputs"]["config_sha1"] == "deadbeef"
    assert data["inputs"]["tokenizer"]["model"] == "m"
    assert data["environment"]["veeksha"]["version"]
    assert data["workload_fingerprint"] is None

    # Second write merges tokenizer / preserves sha1
    write_run_manifest_start(cfg, tokenizer_model="m2")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["inputs"]["config_sha1"] == "deadbeef"
    assert data["inputs"]["tokenizer"]["model"] == "m2"

    finalize_run_manifest(
        str(tmp_path),
        workload_summary={
            "workload_fingerprint": "blake2b:abc",
            "fingerprint_version": 1,
            "sessions": 2,
            "requests": 2,
        },
        outputs={"summary_stats": {"x": 1}},
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["workload_fingerprint"] == "blake2b:abc"
    assert data["sessions"] == 2
    assert data["outputs"]["summary_stats"]["x"] == 1
    assert data["finalized_at"]
