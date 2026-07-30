"""Fetch and publish named benchmark definitions on the Hugging Face Hub.

A definition repo is self-contained: each benchmark lives under
``benchmarks/<name>/`` with its ``benchmark.yml``, pinned config, embedded
assets, and recorded fingerprints. Consumers fetch with
``snapshot_download(allow_patterns=f"benchmarks/{name}/*")`` so a single
benchmark never pulls the whole repository.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from veeksha.logger import init_logger

logger = init_logger(__name__)

DEFAULT_BENCHMARKS_REPO = os.environ.get(
    "VEEKSHA_BENCHMARKS_REPO", "avartha/veeksha-benchmarks"
)


def default_repo() -> str:
    return DEFAULT_BENCHMARKS_REPO


def hub_path_for(name: str) -> str:
    """Return the repo-relative path for a named benchmark."""
    if not name or "/" in name or name in (".", ".."):
        raise ValueError(f"Invalid benchmark name: {name!r}")
    return f"benchmarks/{name}"


def fetch_benchmark(
    name: str,
    *,
    repo: Optional[str] = None,
    revision: Optional[str] = None,
    local_dir: Optional[str | Path] = None,
    force: bool = False,
) -> Path:
    """Download one named benchmark definition into ``local_dir``.

    Returns the local path to ``benchmarks/<name>/``.
    """
    from huggingface_hub import snapshot_download

    repo_id = repo or default_repo()
    rel = hub_path_for(name)
    dest_root = (
        Path(local_dir)
        if local_dir
        else Path(tempfile.mkdtemp(prefix="veeksha-bench-"))
    )
    dest_root.mkdir(parents=True, exist_ok=True)
    target = dest_root / rel
    if target.exists() and not force:
        return target

    staging = Path(tempfile.mkdtemp(dir=dest_root, prefix=".hub_staging_"))
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision or None,
            allow_patterns=[f"{rel}/**", f"{rel}"],
            local_dir=str(staging),
        )
        fetched = staging / rel
        if not fetched.is_dir():
            raise FileNotFoundError(
                f"{rel} not found in {repo_id} " f"(revision: {revision or 'main'})."
            )
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(fetched), str(target))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info("Fetched benchmark %s from %s into %s", name, repo_id, target)
    return target


def publish_benchmark(
    local_dir: str | Path,
    name: str,
    *,
    repo: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
    tag: Optional[str] = None,
) -> str:
    """Upload ``local_dir`` (a benchmarks/<name> tree) to the Hub.

    Returns the repo id that was published to.
    """
    from huggingface_hub import HfApi

    repo_id = repo or default_repo()
    local = Path(local_dir)
    if not local.is_dir():
        raise FileNotFoundError(f"Benchmark directory not found: {local}")
    definition = local / "benchmark.yml"
    if not definition.is_file():
        raise FileNotFoundError(f"Missing benchmark.yml in {local}")

    rel = hub_path_for(name)
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(local),
        path_in_repo=rel,
        commit_message=commit_message or f"Update benchmark {name}",
    )
    if tag:
        api.create_tag(repo_id, tag=tag, repo_type="dataset")
        logger.info("Tagged %s as %s", repo_id, tag)
    logger.info("Published %s -> %s/%s", local, repo_id, rel)
    return repo_id


def load_definition(path: str | Path) -> dict[str, Any]:
    """Load a ``benchmark.yml`` definition as a plain dict.

    Supports vidhi ``!include`` via :func:`vidhi.utils.load_yaml_config`.
    """
    from vidhi.utils import load_yaml_config

    data = load_yaml_config(str(path))
    if not isinstance(data, dict):
        raise ValueError(f"Benchmark definition must be a mapping: {path}")
    return data


def definition_dir(path: str | Path) -> Path:
    """Return the directory that contains the definition and its assets."""
    p = Path(path)
    return p if p.is_dir() else p.parent
