#!/usr/bin/env python3
"""Fetch and publish veeksha voice trace datasets on the Hugging Face Hub.

One dataset repo holds every trace, organized by benchmark direction:

  avartha/veeksha-voice-traces/
    stt/aa_public/        <->  traces/asr/aa_public/
    stt/ami_word_timed/   <->  traces/asr/ami_word_timed/
    tts/<name>/           <->  traces/tts/<name>/

Each dataset directory holds manifest.jsonl, optional filtered variant
manifests (manifest.<variant>.jsonl) that share the same audio pool, the
audio/ tree, and build_info.json provenance written by
prepare_audio_traces.py.

Examples:
  # Download the default STT trace at a pinned revision
  python scripts/audio_trace_hub.py fetch \
      --repo avartha/veeksha-voice-traces --revision v1

  # Download a filtered variant plus the AMI trace
  python scripts/audio_trace_hub.py fetch \
      --repo avartha/veeksha-voice-traces \
      --datasets stt/aa_public,stt/ami_word_timed --variant max15s

  # Publish a locally prepared trace and tag the release
  python scripts/audio_trace_hub.py publish \
      --repo avartha/veeksha-voice-traces --datasets stt/aa_public --tag v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACES_ROOT = REPO_ROOT / "traces"
DEFAULT_REPO = os.environ.get("VEEKSHA_TRACES_REPO", "avartha/veeksha-voice-traces")
DEFAULT_DATASETS = "stt/aa_public"
MANIFEST_NAME = "manifest.jsonl"
BUILD_INFO_NAME = "build_info.json"

# Repo directories are named by benchmark direction; the local checkout
# keeps the pre-existing traces/asr layout the sample configs point at.
LOCAL_DIR_BY_PREFIX = {"stt": "asr", "tts": "tts"}

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def resolve_dataset(spec: str, traces_root: Path = TRACES_ROOT) -> tuple[str, Path]:
    """Map a repo dataset spec like ``stt/aa_public`` to (repo_path, local_dir)."""
    parts = spec.strip().strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise SystemExit(
            f"Dataset must look like 'stt/<name>' or 'tts/<name>', got {spec!r}"
        )
    prefix, name = parts
    if prefix not in LOCAL_DIR_BY_PREFIX:
        raise SystemExit(
            f"Unknown dataset prefix {prefix!r} in {spec!r}. "
            f"Supported: {', '.join(sorted(LOCAL_DIR_BY_PREFIX))}"
        )
    if not _NAME_RE.fullmatch(name):
        raise SystemExit(f"Invalid dataset name {name!r} in {spec!r}")
    return f"{prefix}/{name}", traces_root / LOCAL_DIR_BY_PREFIX[prefix] / name


def manifest_name(variant: str = "") -> str:
    if not variant:
        return MANIFEST_NAME
    if not _NAME_RE.fullmatch(variant):
        raise SystemExit(f"Invalid variant name {variant!r}")
    return f"manifest.{variant}.jsonl"


def list_variants(local_dir: Path) -> list[str]:
    variants = []
    for path in sorted(local_dir.glob("manifest.*.jsonl")):
        variants.append(path.name[len("manifest.") : -len(".jsonl")])
    return variants


def validate_trace_dir(local_dir: Path) -> list[str]:
    """Return a list of problems; empty means the trace directory is sound."""
    problems: list[str] = []
    base_manifest = local_dir / MANIFEST_NAME
    if not base_manifest.is_file():
        return [f"{base_manifest} is missing"]

    for manifest in [base_manifest] + [
        local_dir / manifest_name(v) for v in list_variants(local_dir)
    ]:
        rows = 0
        for line_number, line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"{manifest.name}:{line_number}: invalid JSON ({exc})")
                continue
            audio_file = row.get("audio_file")
            if not audio_file:
                problems.append(f"{manifest.name}:{line_number}: missing audio_file")
            elif not (local_dir / audio_file).is_file():
                problems.append(
                    f"{manifest.name}:{line_number}: audio file not found: "
                    f"{audio_file}"
                )
        if rows == 0:
            problems.append(f"{manifest.name} has no rows")
    return problems


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


def ensure_build_info(local_dir: Path) -> None:
    """Create minimal provenance when a trace predates build_info support."""
    path = local_dir / BUILD_INFO_NAME
    if path.is_file():
        return
    print(
        f"  WARNING: {path} missing; writing minimal publish-time provenance. "
        "Rebuild with prepare_audio_traces.py for full provenance.",
        file=sys.stderr,
    )
    info = {
        "tool": "audio_trace_hub.py",
        "note": "created at publish time; prepare-time provenance unavailable",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "veeksha_git_commit": git_commit(),
    }
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def split_dataset_specs(raw: str) -> list[str]:
    specs = [spec.strip() for spec in raw.split(",") if spec.strip()]
    if not specs:
        raise SystemExit("--datasets must include at least one dataset")
    return specs


def require_repo(args: argparse.Namespace) -> str:
    if not args.repo:
        raise SystemExit(
            "Pass --repo avartha/veeksha-voice-traces " "(or set VEEKSHA_TRACES_REPO)."
        )
    return args.repo


def config_hint_path(manifest: Path) -> str:
    try:
        return manifest.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(manifest)


def run_fetch(args: argparse.Namespace, traces_root: Path = TRACES_ROOT) -> None:
    repo = require_repo(args)
    traces_root.mkdir(parents=True, exist_ok=True)

    for spec in split_dataset_specs(args.datasets):
        repo_path, local_dir = resolve_dataset(spec, traces_root)
        if local_dir.exists() and not args.force:
            raise SystemExit(f"{local_dir} already exists; pass --force to replace it.")

        print(f"Fetching {repo}/{repo_path} (revision: {args.revision or 'main'})")
        staging = Path(tempfile.mkdtemp(dir=traces_root, prefix=".hub_staging_"))
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                revision=args.revision or None,
                allow_patterns=[f"{repo_path}/**"],
                local_dir=str(staging),
            )
            fetched = staging / repo_path
            if not fetched.is_dir():
                raise SystemExit(
                    f"{repo_path} not found in {repo} "
                    f"(revision: {args.revision or 'main'})."
                )
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(fetched), str(local_dir))
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        problems = validate_trace_dir(local_dir)
        if problems:
            raise SystemExit(
                f"Fetched trace {spec} failed validation:\n  " + "\n  ".join(problems)
            )

        manifest = local_dir / manifest_name(args.variant)
        if not manifest.is_file():
            available = list_variants(local_dir)
            raise SystemExit(
                f"Variant {args.variant!r} not found in {spec}. Available "
                f"variants: {', '.join(available) if available else '(none)'}"
            )
        print(f"  Ready. Point trace_file at: {config_hint_path(manifest)}")


def run_publish(args: argparse.Namespace, traces_root: Path = TRACES_ROOT) -> None:
    repo = require_repo(args)
    api = HfApi()

    for spec in split_dataset_specs(args.datasets):
        repo_path, local_dir = resolve_dataset(spec, traces_root)
        if not local_dir.is_dir():
            raise SystemExit(
                f"{local_dir} does not exist; build it with "
                "prepare_audio_traces.py first."
            )
        problems = validate_trace_dir(local_dir)
        if problems:
            raise SystemExit(f"Refusing to publish {spec}:\n  " + "\n  ".join(problems))
        ensure_build_info(local_dir)

        print(f"Publishing {local_dir} -> {repo}/{repo_path}")
        api.create_repo(repo, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(local_dir),
            path_in_repo=repo_path,
            commit_message=args.commit_message or f"Update {repo_path}",
        )

    if args.tag:
        api.create_tag(repo, tag=args.tag, repo_type="dataset")
        print(f"Tagged {repo} as {args.tag}")
    print(f"Done: https://huggingface.co/datasets/{repo}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--repo",
            default=DEFAULT_REPO,
            help=(
                "Hugging Face dataset repo id, e.g. avartha/veeksha-voice-traces. "
                "Defaults to $VEEKSHA_TRACES_REPO."
            ),
        )
        sub.add_argument(
            "--datasets",
            default=DEFAULT_DATASETS,
            help=(
                "Comma-separated dataset specs (<prefix>/<name> with prefix "
                f"stt or tts). Default: {DEFAULT_DATASETS}."
            ),
        )

    fetch = subparsers.add_parser("fetch", help="Download traces from the Hub.")
    add_shared(fetch)
    fetch.add_argument(
        "--revision",
        default="",
        help="Repo revision (tag, branch, or commit) to pin. Default: main.",
    )
    fetch.add_argument(
        "--variant",
        default="",
        help="Filtered manifest variant to verify (manifest.<variant>.jsonl).",
    )
    fetch.add_argument(
        "--force",
        action="store_true",
        help="Replace existing local trace directories.",
    )

    publish = subparsers.add_parser("publish", help="Upload traces to the Hub.")
    add_shared(publish)
    publish.add_argument(
        "--tag",
        default="",
        help="Create this tag on the repo after uploading (e.g. v1).",
    )
    publish.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist yet.",
    )
    publish.add_argument(
        "--commit-message",
        default="",
        help="Commit message for the upload.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "fetch":
        run_fetch(args)
    elif args.command == "publish":
        run_publish(args)


if __name__ == "__main__":
    main()
