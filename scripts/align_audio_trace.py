#!/usr/bin/env python3
"""Add reference word timestamps to an ASR trace manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_NEMO_MODEL = "stt_en_fastconformer_hybrid_large_pc"
NEMO_MANIFEST_NAME = "nemo_manifest.jsonl"
NEMO_OUTPUT_DIR = "nemo_output"
STAGED_AUDIO_DIR = "staged_audio"


@dataclass
class AlignmentItem:
    row_index: int
    audio_path: Path
    text: str


@dataclass
class AlignmentPlan:
    rows: list[dict[str, Any]]
    items: list[AlignmentItem]
    nemo_manifest: Path
    nemo_output_dir: Path


@dataclass(frozen=True)
class WordTiming:
    word: str
    start_ms: float
    end_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input ASR JSONL manifest.")
    parser.add_argument(
        "--output-manifest",
        required=True,
        help="Output JSONL manifest with reference_word_timestamps.",
    )
    parser.add_argument(
        "--alignment-output-dir",
        required=True,
        help="Directory for intermediate NeMo manifests, CTM files, and staged audio.",
    )
    parser.add_argument(
        "--nemo-align-script",
        default=os.environ.get("NEMO_ALIGN_SCRIPT", ""),
        help="Path to NeMo tools/nemo_forced_aligner/align.py.",
    )
    return parser.parse_args()


def align_manifest(
    *,
    manifest: str | Path,
    output_manifest: str | Path,
    alignment_output_dir: str | Path,
    nemo_align_script: str | Path,
) -> Path:
    manifest_path = Path(manifest)
    output_path = Path(output_manifest)
    work_dir = Path(alignment_output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if not str(nemo_align_script):
        raise SystemExit(
            "--nemo-align-script or NEMO_ALIGN_SCRIPT is required for NeMo alignment."
        )

    plan = build_alignment_plan(manifest_path, work_dir)
    write_nemo_manifest(plan.items, plan.nemo_manifest)
    NeMoForcedAligner(Path(nemo_align_script)).run(
        plan.nemo_manifest, plan.nemo_output_dir
    )

    aligned_rows = attach_word_timings(plan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, aligned_rows)
    return output_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_alignment_plan(manifest_path: Path, work_dir: Path) -> AlignmentPlan:
    rows = read_jsonl(manifest_path)
    items = build_alignment_items(
        rows,
        manifest_path.parent,
        work_dir / STAGED_AUDIO_DIR,
    )
    return AlignmentPlan(
        rows=rows,
        items=items,
        nemo_manifest=work_dir / NEMO_MANIFEST_NAME,
        nemo_output_dir=work_dir / NEMO_OUTPUT_DIR,
    )


def build_alignment_items(
    rows: list[dict[str, Any]], manifest_dir: Path, staging_dir: Path
) -> list[AlignmentItem]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    items: list[AlignmentItem] = []

    for row_index, row in enumerate(rows):
        audio_path = stage_audio(
            resolve_audio_path(row, manifest_dir),
            staging_dir / f"row_{row_index:06d}.wav",
        )
        items.append(
            AlignmentItem(
                row_index=row_index,
                audio_path=audio_path,
                text=str(row["expected_transcript"]),
            )
        )

    return items


def stage_audio(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.symlink(source, target)
    except OSError:
        shutil.copyfile(source, target)
    return target


def resolve_audio_path(row: dict[str, Any], manifest_dir: Path) -> Path:
    audio_path = Path(str(row["audio_file"]))
    if not audio_path.is_absolute():
        audio_path = manifest_dir / audio_path
    return audio_path.resolve()


def write_nemo_manifest(items: list[AlignmentItem], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(
                json.dumps(
                    {
                        "audio_filepath": str(item.audio_path.resolve()),
                        "text": item.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


class NeMoForcedAligner:
    """Runs NeMo forced alignment with the repository's fixed defaults."""

    def __init__(self, align_script: Path) -> None:
        self.align_script = align_script

    def run(self, manifest_path: Path, output_dir: Path) -> None:
        if not self.align_script.exists():
            raise FileNotFoundError(f"NeMo align.py not found: {self.align_script}")

        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(self.command(manifest_path, output_dir), check=True)

    def command(self, manifest_path: Path, output_dir: Path) -> list[str]:
        return [
            sys.executable,
            str(self.align_script),
            f"manifest_filepath={manifest_path}",
            f"output_dir={output_dir}",
            'save_output_file_formats=["ctm"]',
            f"pretrained_name={DEFAULT_NEMO_MODEL}",
        ]


def attach_word_timings(plan: AlignmentPlan) -> list[dict[str, Any]]:
    output_manifest = find_nemo_output_manifest(
        plan.nemo_output_dir,
        plan.nemo_manifest,
    )
    nemo_rows = read_jsonl(output_manifest)
    if len(nemo_rows) != len(plan.items):
        raise ValueError(
            f"NeMo output row count {len(nemo_rows)} does not match input "
            f"{len(plan.items)}."
        )

    aligned_rows = [dict(row) for row in plan.rows]
    for item, nemo_row in zip(plan.items, nemo_rows):
        ctm_path = find_word_ctm_path(nemo_row, output_manifest.parent)
        words = parse_word_ctm(ctm_path)
        aligned_rows[item.row_index]["reference_word_timestamps"] = [
            {
                "word": word.word,
                "start_ms": round(word.start_ms, 3),
                "end_ms": round(word.end_ms, 3),
            }
            for word in words
        ]
    return aligned_rows


def find_nemo_output_manifest(nemo_output_dir: Path, nemo_manifest: Path) -> Path:
    stem = nemo_manifest.stem
    candidates = [
        nemo_output_dir / f"{stem}_with_output_file_paths.json",
        nemo_output_dir / f"{stem}_with_ctm_paths.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(nemo_output_dir.glob(f"{stem}_with*.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find NeMo output manifest in {nemo_output_dir}")


def find_word_ctm_path(row: dict[str, Any], base_dir: Path) -> Path:
    for key, value in row.items():
        if "word" in key and "ctm" in key and value:
            path = Path(str(value))
            return path if path.is_absolute() else base_dir / path
    raise ValueError(f"NeMo row has no word-level CTM path: {row}")


def parse_word_ctm(path: Path) -> list[WordTiming]:
    words: list[WordTiming] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=4)
            if len(parts) < 5:
                continue
            start_s = float(parts[2])
            duration_s = float(parts[3])
            word = parts[4]
            words.append(
                WordTiming(
                    word=word,
                    start_ms=start_s * 1000,
                    end_ms=(start_s + duration_s) * 1000,
                )
            )
    return words


def main() -> None:
    args = parse_args()
    align_manifest(
        manifest=args.manifest,
        output_manifest=args.output_manifest,
        alignment_output_dir=args.alignment_output_dir,
        nemo_align_script=args.nemo_align_script,
    )


if __name__ == "__main__":
    main()
