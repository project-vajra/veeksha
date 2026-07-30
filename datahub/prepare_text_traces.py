#!/usr/bin/env python3
"""Build portable text (TTS) trace pools.

Each pool lands at traces/tts/<name> with a flat text manifest.jsonl
(rows: session_id, sample_id, dataset, text) consumable by veeksha's
seed_tts_text flavor (local_path + text_column/id_column) and by
`trace_hub.py mix`. The sharegpt pool additionally keeps the native
conversation file for the sharegpt flavor's role/filter knobs.

Examples:
  python datahub/prepare_text_traces.py --datasets seed_tts_en

  python datahub/prepare_text_traces.py --datasets sharegpt \
    --sharegpt-source /path/to/sharegpt_data.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACES_ROOT = REPO_ROOT / "traces"
MANIFEST_NAME = "manifest.jsonl"
BUILD_INFO_NAME = "build_info.json"

SEED_TTS_DATASET = "TwinkStart/Seed-TTS-Eval"
SEED_TTS_SUBSET = "en"
SEED_TTS_SPLIT = "train"
SHAREGPT_NATIVE_NAME = "sharegpt_data.json"
SHAREGPT_ASSISTANT_ROLE = "gpt"

DATASETS = ("seed_tts_en", "sharegpt")


def git_commit(repo_root: Path = REPO_ROOT) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_manifest(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_build_info(output_dir: Path, extra: dict[str, Any]) -> Path:
    info = {
        "tool": "prepare_text_traces.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "veeksha_git_commit": git_commit(),
        "argv": sys.argv[1:],
        **extra,
    }
    path = output_dir / BUILD_INFO_NAME
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return path


def flatten_sharegpt_conversations(
    conversations: list[dict[str, Any]],
    assistant_role: str = SHAREGPT_ASSISTANT_ROLE,
) -> Iterable[dict[str, Any]]:
    """One manifest row per assistant turn; no length/quality filtering.

    The pool is the full corpus — filtering (alpha ratio, length windows)
    stays a generation-time knob on the trace flavors.
    """
    session_id = 0
    for conversation in conversations:
        conversation_id = conversation.get("id", "")
        for turn_index, turn in enumerate(conversation.get("conversations", [])):
            if turn.get("from") != assistant_role:
                continue
            text = str(turn.get("value") or "").strip()
            if not text:
                continue
            yield {
                "session_id": session_id,
                "sample_id": f"{conversation_id}:{turn_index}",
                "dataset": "sharegpt",
                "text": text,
            }
            session_id += 1


def build_seed_tts_en(output_dir: Path) -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    dataset = load_dataset(SEED_TTS_DATASET, SEED_TTS_SUBSET, split=SEED_TTS_SPLIT)
    # Keep only the text columns; the dataset also carries voice-cloning
    # prompt audio, which would otherwise be decoded on iteration.
    dataset = dataset.select_columns(["filename", "text"])
    rows = (
        {
            "session_id": index,
            "sample_id": str(row["filename"]),
            "dataset": "seed_tts_en",
            "text": str(row["text"]),
        }
        for index, row in enumerate(dataset)
    )
    count = write_manifest(output_dir / MANIFEST_NAME, rows)

    try:
        revision = HfApi().dataset_info(SEED_TTS_DATASET).sha
    except Exception:
        revision = None
    write_build_info(
        output_dir,
        {
            "datasets": {
                "seed_tts_en": {
                    "repo": SEED_TTS_DATASET,
                    "subset": SEED_TTS_SUBSET,
                    "split": SEED_TTS_SPLIT,
                }
            },
            "source_revisions": {"seed_tts_en": revision},
            "clip_count": count,
        },
    )
    print(f"seed_tts_en: {count} rows -> {output_dir / MANIFEST_NAME}")


def build_sharegpt(output_dir: Path, source: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"ShareGPT source not found: {source}")

    native_path = output_dir / SHAREGPT_NATIVE_NAME
    if not native_path.exists():
        shutil.copyfile(source, native_path)

    with native_path.open("r", encoding="utf-8") as f:
        conversations = json.load(f)
    count = write_manifest(
        output_dir / MANIFEST_NAME,
        flatten_sharegpt_conversations(conversations),
    )

    write_build_info(
        output_dir,
        {
            "datasets": {"sharegpt": {"source_file": str(source)}},
            "source_sha256": {SHAREGPT_NATIVE_NAME: file_sha256(native_path)},
            "conversation_count": len(conversations),
            "clip_count": count,
            "assistant_role": SHAREGPT_ASSISTANT_ROLE,
            "note": (
                "manifest.jsonl flattens assistant turns with no filtering; "
                f"{SHAREGPT_NATIVE_NAME} keeps the native conversations for "
                "the sharegpt trace flavor"
            ),
        },
    )
    print(f"sharegpt: {count} assistant turns -> {output_dir / MANIFEST_NAME}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="seed_tts_en",
        help=f"Comma-separated subset to build. Supported: {', '.join(DATASETS)}.",
    )
    parser.add_argument(
        "--sharegpt-source",
        default="",
        help="Path to a ShareGPT-format JSON file (required for sharegpt).",
    )
    args = parser.parse_args()

    keys = [key.strip() for key in args.datasets.split(",") if key.strip()]
    unknown = [key for key in keys if key not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset key(s): {unknown}. Supported: {DATASETS}")

    for key in keys:
        output_dir = TRACES_ROOT / "tts" / key
        output_dir.mkdir(parents=True, exist_ok=True)
        if key == "seed_tts_en":
            build_seed_tts_en(output_dir)
        elif key == "sharegpt":
            if not args.sharegpt_source:
                raise SystemExit("sharegpt requires --sharegpt-source")
            build_sharegpt(output_dir, Path(args.sharegpt_source))


if __name__ == "__main__":
    main()
