#!/usr/bin/env python3
"""Fetch, compose, and publish trace datasets on the Hugging Face Hub.

One dataset repo holds every trace, organized by benchmark direction:

  avartha/veeksha-voice-traces/
    stt/aa_voxpopuli/     <->  traces/asr/aa_voxpopuli/     (pool)
    stt/aa_earnings22/    <->  traces/asr/aa_earnings22/    (pool)
    stt/ami_word_timed_2k/ <-> traces/asr/ami_word_timed_2k/ (subset)
    stt/aa_bench100/         <->  traces/asr/aa_bench100/         (mixture)
    tts/<name>/           <->  traces/tts/<name>/

Pools are source-pure and carry the audio; mixtures compose pools (or other
mixtures) via a manifest whose rows reference sibling pool audio, plus a
mixture.json recipe with flattened pool dependencies. Every pool carries
build_info.json provenance written by prepare_audio_traces.py.

Examples:
  # Download the benchmark mixture (pool dependencies fetched automatically)
  python datahub/trace_hub.py fetch --datasets stt/aa_bench100 --revision v0.1

  # Compose a mixture: 50 seeded picks from each pool
  python datahub/trace_hub.py mix --name aa_bench100 \
      --take stt/aa_voxpopuli:50 --take stt/aa_earnings22:50 --seed 42

  # Slice a corpus too large to host in full into a standalone pool
  python datahub/trace_hub.py subset --dataset stt/ami_word_timed \
      --name ami_word_timed_2k --count 2000 --seed 42

  # Publish pools and tag the release
  python datahub/trace_hub.py publish \
      --datasets stt/aa_voxpopuli,stt/aa_earnings22 --tag v0.1
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
DEFAULT_DATASETS = "stt/aa_bench100"
MANIFEST_NAME = "manifest.jsonl"
BUILD_INFO_NAME = "build_info.json"
MIXTURE_INFO_NAME = "mixture.json"

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
    """Return a list of problems; empty means the trace directory is sound.

    Mixture manifests may reference sibling pools (``../<pool>/audio/...``);
    anything resolving outside the datasets root is rejected.
    """
    problems: list[str] = []
    base_manifest = local_dir / MANIFEST_NAME
    if not base_manifest.is_file():
        return [f"{base_manifest} is missing"]
    datasets_root = local_dir.resolve().parent

    for manifest in [base_manifest] + [
        local_dir / manifest_name(v) for v in list_variants(local_dir)
    ]:
        rows = 0
        for line_number, line in enumerate(
            # JSONL is delimited by \n only; str.splitlines() would also
            # split on U+2028/U+2029 inside JSON string values.
            manifest.read_text(encoding="utf-8").split("\n"),
            start=1,
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
                # Text rows (TTS pools) carry their payload inline.
                if not str(row.get("text") or "").strip():
                    problems.append(
                        f"{manifest.name}:{line_number}: row has neither "
                        "audio_file nor text"
                    )
                continue
            resolved = (local_dir / audio_file).resolve()
            if not resolved.is_relative_to(datasets_root):
                problems.append(
                    f"{manifest.name}:{line_number}: audio path escapes the "
                    f"datasets root: {audio_file}"
                )
            elif not resolved.is_file():
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
    if (local_dir / MIXTURE_INFO_NAME).is_file():
        return  # mixtures carry their provenance in mixture.json
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


def _fetch_one(spec: str, args: argparse.Namespace, traces_root: Path) -> Path:
    repo = require_repo(args)
    repo_path, local_dir = resolve_dataset(spec, traces_root)

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
    return local_dir


def mixture_requires(local_dir: Path) -> list[str]:
    mixture_info = local_dir / MIXTURE_INFO_NAME
    if not mixture_info.is_file():
        return []
    return list(json.loads(mixture_info.read_text()).get("requires", []))


def run_fetch(args: argparse.Namespace, traces_root: Path = TRACES_ROOT) -> None:
    require_repo(args)
    traces_root.mkdir(parents=True, exist_ok=True)

    requested = split_dataset_specs(args.datasets)
    for spec in requested:
        _, local_dir = resolve_dataset(spec, traces_root)
        if local_dir.exists() and not args.force:
            raise SystemExit(f"{local_dir} already exists; pass --force to replace it.")

    fetched_dirs: list[tuple[str, Path]] = []
    queue = list(requested)
    done: set[str] = set()
    while queue:
        spec = queue.pop(0)
        if spec in done:
            continue
        done.add(spec)
        _, local_dir = resolve_dataset(spec, traces_root)
        # Dependencies that already exist locally are kept as-is; only the
        # explicitly requested specs honor --force replacement.
        if spec not in requested and local_dir.exists():
            continue
        local_dir = _fetch_one(spec, args, traces_root)
        fetched_dirs.append((spec, local_dir))
        for dependency in mixture_requires(local_dir):
            if dependency not in done:
                print(f"  {spec} requires {dependency}")
                queue.append(dependency)

    # Validate after all dependencies are in place (mixtures reference
    # sibling pool audio).
    for spec, local_dir in fetched_dirs:
        problems = validate_trace_dir(local_dir)
        if problems:
            raise SystemExit(
                f"Fetched trace {spec} failed validation:\n  " + "\n  ".join(problems)
            )

    for spec in requested:
        _, local_dir = resolve_dataset(spec, traces_root)
        manifest = local_dir / manifest_name(args.variant)
        if not manifest.is_file():
            available = list_variants(local_dir)
            raise SystemExit(
                f"Variant {args.variant!r} not found in {spec}. Available "
                f"variants: {', '.join(available) if available else '(none)'}"
            )
        print(f"  Ready. Point trace_file at: {config_hint_path(manifest)}")


def row_key(row: dict) -> str:
    return str(row.get("sample_id") or row.get("source_id") or "")


def load_manifest_rows(path: Path) -> list[dict]:
    rows = []
    # \n-delimited only: splitlines() would split on U+2028 inside values.
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_variant(args: argparse.Namespace, traces_root: Path = TRACES_ROOT) -> None:
    """Derive manifest.<name>.jsonl by matching a reference manifest's clips
    (by sample_id) against the local pool, so the variant reuses pool audio."""
    spec = args.dataset
    _, local_dir = resolve_dataset(spec, traces_root)
    pool_manifest = local_dir / MANIFEST_NAME
    if not pool_manifest.is_file():
        raise SystemExit(f"{pool_manifest} not found; fetch or build {spec} first.")
    reference_path = Path(args.reference)
    if not reference_path.is_file():
        raise SystemExit(f"Reference manifest not found: {reference_path}")
    target = local_dir / manifest_name(args.name)
    if target.exists() and not args.force:
        raise SystemExit(f"{target} already exists; pass --force to replace it.")

    pool_by_key: dict[str, dict] = {}
    duplicates = set()
    for row in load_manifest_rows(pool_manifest):
        key = row_key(row)
        if key in pool_by_key:
            duplicates.add(key)
        pool_by_key[key] = row
    if duplicates:
        raise SystemExit(
            f"Pool manifest has {len(duplicates)} duplicate sample ids "
            f"(e.g. {sorted(duplicates)[:3]}); cannot match reliably."
        )

    reference_rows = load_manifest_rows(reference_path)
    missing = [
        row_key(row) for row in reference_rows if row_key(row) not in pool_by_key
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(reference_rows)} reference clips are not in "
            f"the {spec} pool (e.g. {missing[:3]}). The pool build may "
            "predate or postdate the reference; rebuild or adjust."
        )

    transcript_drift = 0
    variant_rows = []
    for index, reference_row in enumerate(reference_rows):
        pool_row = dict(pool_by_key[row_key(reference_row)])
        if pool_row.get("expected_transcript") != reference_row.get(
            "expected_transcript"
        ):
            transcript_drift += 1
        pool_row["session_id"] = index
        variant_rows.append(pool_row)

    with target.open("w", encoding="utf-8") as f:
        for row in variant_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if transcript_drift:
        print(
            f"  WARNING: {transcript_drift}/{len(variant_rows)} clips have a "
            "different transcript in the pool than in the reference manifest "
            "(upstream data drift?). The pool's transcript was kept.",
            file=sys.stderr,
        )
    print(f"Wrote {len(variant_rows)} clips -> {target}")
    print(f"  Point trace_file at: {config_hint_path(target)}")


def parse_take(raw: str) -> tuple[str, int]:
    spec, sep, count = raw.rpartition(":")
    if not sep or not count.isdigit() or int(count) < 1:
        raise SystemExit(f"--take must look like 'stt/<name>:<count>', got {raw!r}")
    return spec, int(count)


def load_mix_source(spec: str, traces_root: Path) -> tuple[list[dict], list[str]]:
    """Load a mix source's rows (paths valid from any sibling dir) and the
    pool specs it transitively requires."""
    _, source_dir = resolve_dataset(spec, traces_root)
    manifest = source_dir / MANIFEST_NAME
    if not manifest.is_file():
        raise SystemExit(f"{manifest} not found; fetch or build {spec} first.")
    rows = load_manifest_rows(manifest)

    mixture_info = source_dir / MIXTURE_INFO_NAME
    if mixture_info.is_file():
        # A mixture's rows already use sibling-relative paths, which are
        # position-independent among datasets in the same root; its
        # dependencies are its own (already flattened) pool requirements.
        requires = list(json.loads(mixture_info.read_text())["requires"])
        return rows, requires

    # Pool audio rows are pool-relative; rewrite to sibling-relative.
    # Text rows carry their payload inline and need no rewriting.
    rewritten = []
    for row in rows:
        if row.get("audio_file"):
            row = dict(row)
            row["audio_file"] = f"../{source_dir.name}/{row['audio_file']}"
        rewritten.append(row)
    return rewritten, [spec]


def run_mix(args: argparse.Namespace, traces_root: Path = TRACES_ROOT) -> None:
    """Compose a mixture dataset from pools and/or other mixtures."""
    if bool(args.take) == bool(args.reference):
        raise SystemExit("mix needs exactly one of --take ... or --reference.")

    if args.take:
        takes = [parse_take(raw) for raw in args.take]
        source_specs = [spec for spec, _ in takes]
    else:
        source_specs = split_dataset_specs(args.sources)
        if not args.sources:
            raise SystemExit("--reference mode needs --sources <spec,...>.")
        takes = []

    prefixes = {spec.split("/")[0] for spec in source_specs}
    if len(prefixes) != 1:
        raise SystemExit(f"Mix sources must share one prefix, got {prefixes}.")
    out_spec = f"{next(iter(prefixes))}/{args.name}"
    _, out_dir = resolve_dataset(out_spec, traces_root)
    if (out_dir / MANIFEST_NAME).exists() and not args.force:
        raise SystemExit(f"{out_dir} already exists; pass --force to replace it.")

    sources = {spec: load_mix_source(spec, traces_root) for spec in source_specs}
    requires: list[str] = []
    for _, source_requires in sources.values():
        for req in source_requires:
            if req not in requires:
                requires.append(req)

    seen: set[str] = set()
    selected: list[dict] = []

    if takes:
        rng = random.Random(args.seed)
        for spec, count in takes:
            rows, _ = sources[spec]
            if not args.allow_duplicates:
                rows = [row for row in rows if row_key(row) not in seen]
            if len(rows) < count:
                raise SystemExit(
                    f"{spec} has only {len(rows)} un-selected clips, "
                    f"cannot take {count}."
                )
            for row in rng.sample(rows, count):
                seen.add(row_key(row))
                selected.append(row)
    else:
        by_key: dict[str, dict] = {}
        for spec, (rows, _) in sources.items():
            for row in rows:
                key = row_key(row)
                if key in by_key:
                    raise SystemExit(
                        f"sample id {key!r} appears in multiple sources; "
                        "cannot match a reference against them."
                    )
                by_key[key] = row
        reference_path = Path(args.reference)
        if not reference_path.is_file():
            raise SystemExit(f"Reference manifest not found: {reference_path}")
        for row in load_manifest_rows(reference_path):
            key = row_key(row)
            if key not in by_key:
                raise SystemExit(
                    f"Reference clip {key!r} not found in sources " f"{source_specs}."
                )
            if key in seen and not args.allow_duplicates:
                continue
            seen.add(key)
            selected.append(by_key[key])

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / MANIFEST_NAME).open("w", encoding="utf-8") as f:
        for index, row in enumerate(selected):
            row = dict(row)
            row["session_id"] = index
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mixture_info = {
        "name": args.name,
        "requires": requires,
        "recipe": {
            "mode": "take" if takes else "reference",
            "takes": [{"source": s, "count": c} for s, c in takes],
            "reference": args.reference,
            "sources": source_specs,
            "seed": args.seed,
            "allow_duplicates": args.allow_duplicates,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "veeksha_git_commit": git_commit(),
    }
    (out_dir / MIXTURE_INFO_NAME).write_text(
        json.dumps(mixture_info, indent=2) + "\n", encoding="utf-8"
    )

    problems = validate_trace_dir(out_dir)
    if problems:
        raise SystemExit(
            f"Mixture {out_spec} failed validation (are its pools present?):"
            "\n  " + "\n  ".join(problems)
        )
    print(f"Wrote {len(selected)} clips -> {out_dir / MANIFEST_NAME}")
    print(f"  Requires: {', '.join(requires)}")
    print(f"  Point trace_file at: {config_hint_path(out_dir / MANIFEST_NAME)}")


def run_subset(args: argparse.Namespace, traces_root: Path = TRACES_ROOT) -> None:
    """Materialize a seeded sample of a pool as a standalone sibling pool.

    Distinct from ``mix``: a mixture's rows point at sibling pool audio, so it
    only makes sense alongside its full parent. A subset copies (hardlinks) the
    audio it selects, so it stands on its own and can be published without the
    parent — the point being to publish a benchmark-sized slice of a corpus
    that is too large to host in full.
    """
    spec = args.dataset
    _, source_dir = resolve_dataset(spec, traces_root)
    source_manifest = source_dir / MANIFEST_NAME
    if not source_manifest.is_file():
        raise SystemExit(f"{source_manifest} not found; fetch or build {spec} first.")
    if (source_dir / MIXTURE_INFO_NAME).is_file():
        raise SystemExit(
            f"{spec} is a mixture, not a pool. Subset its pools instead, or "
            "recompose with `mix --take`."
        )

    out_spec = f"{spec.split('/')[0]}/{args.name}"
    _, out_dir = resolve_dataset(out_spec, traces_root)
    if out_dir.resolve() == source_dir.resolve():
        raise SystemExit("--name must differ from the source dataset.")
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} already exists; pass --force to replace it.")

    rows = load_manifest_rows(source_manifest)
    if args.count > len(rows):
        raise SystemExit(
            f"{spec} has only {len(rows)} clips, cannot subset {args.count}."
        )

    # Sample by index, then restore manifest order: the selection is what the
    # seed fixes, not the ordering, and original order keeps the manifest
    # readable and diffable against the parent.
    rng = random.Random(args.seed)
    picked = sorted(rng.sample(range(len(rows)), args.count))
    selected = [rows[index] for index in picked]

    if out_dir.exists():
        if not out_dir.resolve().is_relative_to(traces_root.resolve()):
            raise SystemExit(f"Refusing to replace {out_dir}: outside {traces_root}.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hardlink where possible: same filesystem, so a subset costs no extra
    # disk and cannot drift from the parent's bytes.
    linked = copied = 0
    for row in selected:
        audio_file = row.get("audio_file")
        if not audio_file:
            continue  # text pools (TTS) carry their payload inline
        source_audio = (source_dir / audio_file).resolve()
        if not source_audio.is_relative_to(source_dir.resolve()):
            raise SystemExit(
                f"{spec} row {row_key(row)!r} points outside the pool "
                f"({audio_file}); only self-contained pools can be subset."
            )
        destination = out_dir / audio_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_audio, destination)
            linked += 1
        except OSError:
            shutil.copy2(source_audio, destination)
            copied += 1

    with (out_dir / MANIFEST_NAME).open("w", encoding="utf-8") as f:
        for index, row in enumerate(selected):
            row = dict(row)
            row["session_id"] = index
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    parent_info = None
    parent_info_path = source_dir / BUILD_INFO_NAME
    if parent_info_path.is_file():
        parent_info = json.loads(parent_info_path.read_text(encoding="utf-8"))
    build_info = {
        "tool": "trace_hub.py subset",
        "argv": sys.argv[1:],
        "subset_of": spec,
        "subset": {
            "count": args.count,
            "seed": args.seed,
            "parent_clip_count": len(rows),
        },
        # Upstream provenance is inherited: a subset introduces no new source.
        "parent_build_info": parent_info,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "veeksha_git_commit": git_commit(),
    }
    (out_dir / BUILD_INFO_NAME).write_text(
        json.dumps(build_info, indent=2) + "\n", encoding="utf-8"
    )

    problems = validate_trace_dir(out_dir)
    if problems:
        raise SystemExit(
            f"Subset {out_spec} failed validation:\n  " + "\n  ".join(problems)
        )
    print(f"Wrote {len(selected)} of {len(rows)} clips -> {out_dir / MANIFEST_NAME}")
    print(f"  Audio: {linked} hardlinked, {copied} copied")
    print(f"  Point trace_file at: {config_hint_path(out_dir / MANIFEST_NAME)}")


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
        # private only applies at creation; publishing never flips an
        # existing repo's visibility.
        api.create_repo(
            repo, repo_type="dataset", private=not args.public, exist_ok=True
        )
        api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(local_dir),
            path_in_repo=repo_path,
            commit_message=args.commit_message or f"Update {repo_path}",
            # NeMo alignment intermediates are build scratch, not dataset.
            ignore_patterns=["alignment/**"],
        )

    if args.tag:
        api.create_tag(repo, tag=args.tag, repo_type="dataset")
        print(f"Tagged {repo} as {args.tag}")
    print(f"Done: https://huggingface.co/datasets/{repo}")


CARDS_DIR = REPO_ROOT / "datahub" / "cards"


def card_path_for_repo(repo: str) -> Path:
    """Each Hub repo's card lives at datahub/cards/<repo-name>.md."""
    return CARDS_DIR / f"{repo.rsplit('/', 1)[-1]}.md"


def run_card(args: argparse.Namespace) -> None:
    repo = require_repo(args)
    card_path = Path(args.file) if args.file else card_path_for_repo(repo)
    if not card_path.is_file():
        raise SystemExit(f"Dataset card not found: {card_path}")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        commit_message=args.commit_message or "Update dataset card",
    )
    print(f"Uploaded dataset card -> https://huggingface.co/datasets/{repo}")


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

    variant = subparsers.add_parser(
        "variant",
        help="Derive a filtered manifest variant from a reference manifest.",
    )
    variant.add_argument(
        "--dataset",
        default=DEFAULT_DATASETS,
        help=f"Dataset spec holding the pool. Default: {DEFAULT_DATASETS}.",
    )
    variant.add_argument(
        "--name",
        required=True,
        help="Variant name; writes manifest.<name>.jsonl next to the pool.",
    )
    variant.add_argument(
        "--reference",
        required=True,
        help=(
            "Path to a reference manifest whose clips (matched by sample_id) "
            "define the variant's membership and order."
        ),
    )
    variant.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing variant manifest.",
    )

    mix = subparsers.add_parser(
        "mix",
        help="Compose a mixture dataset from pools and/or other mixtures.",
    )
    mix.add_argument(
        "--name",
        required=True,
        help="Mixture name; creates <prefix>/<name> with its own manifest.",
    )
    mix.add_argument(
        "--take",
        action="append",
        default=[],
        metavar="SPEC:COUNT",
        help=(
            "Sample COUNT clips from SPEC (pool or mixture), seeded. "
            "Repeatable; order is preserved."
        ),
    )
    mix.add_argument(
        "--reference",
        default="",
        help=(
            "Reference manifest whose clips (matched by sample_id across "
            "--sources) define membership and order."
        ),
    )
    mix.add_argument(
        "--sources",
        default="",
        help="Comma-separated source specs for --reference mode.",
    )
    mix.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for --take sampling. Default: 42.",
    )
    mix.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Permit the same clip more than once (intentional oversampling).",
    )
    mix.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing mixture.",
    )

    subset = subparsers.add_parser(
        "subset",
        help="Materialize a seeded sample of a pool as a standalone pool.",
    )
    subset.add_argument(
        "--dataset",
        required=True,
        help="Source pool spec, e.g. stt/ami_word_timed.",
    )
    subset.add_argument(
        "--name",
        required=True,
        help=(
            "Output name; creates <prefix>/<name>. Name it for the size so it "
            "is not mistaken for the full corpus (e.g. ami_word_timed_2k)."
        ),
    )
    subset.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of clips to sample.",
    )
    subset.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the sample. Default: 42.",
    )
    subset.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing subset directory.",
    )

    publish = subparsers.add_parser("publish", help="Upload traces to the Hub.")
    add_shared(publish)
    publish.add_argument(
        "--tag",
        default="",
        help="Create this tag on the repo after uploading (e.g. v1).",
    )
    publish.add_argument(
        "--public",
        action="store_true",
        help=(
            "Create the repo as public if it does not exist yet " "(default: private)."
        ),
    )
    publish.add_argument(
        "--commit-message",
        default="",
        help="Commit message for the upload.",
    )

    card = subparsers.add_parser(
        "card", help="Upload the dataset card (README.md) to the Hub repo."
    )
    card.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="Hugging Face dataset repo id. Defaults to $VEEKSHA_TRACES_REPO.",
    )
    card.add_argument(
        "--file",
        default="",
        help="Path to the card source. Default: datahub/cards/<repo-name>.md.",
    )
    card.add_argument(
        "--commit-message",
        default="",
        help="Commit message for the card upload.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "fetch":
        run_fetch(args)
    elif args.command == "variant":
        run_variant(args)
    elif args.command == "mix":
        run_mix(args)
    elif args.command == "subset":
        run_subset(args)
    elif args.command == "publish":
        run_publish(args)
    elif args.command == "card":
        run_card(args)


if __name__ == "__main__":
    main()
