#!/usr/bin/env python3
"""Run the deployment-claim workload study search."""

import os
from pathlib import Path
import sys


MODULE = "veeksha.case_studies.workload_deployment_claim_search"


def _is_veeksha_repo(path: Path) -> bool:
    return (path / "veeksha" / "case_studies").exists() and (path / "pyproject.toml").exists()


def _find_veeksha_repo() -> Path | None:
    script_path = Path(__file__).resolve()
    candidates: list[Path] = []

    env_repo = os.environ.get("VEEKSHA_REPO")
    if env_repo:
        candidates.append(Path(env_repo).expanduser().resolve())

    for parent in script_path.parents:
        candidates.append(parent)
        candidates.append(parent / "veeksha")
        candidates.append(parent / "veeksha-prs")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_veeksha_repo(candidate):
            return candidate
    return None


def _prepend_pythonpath(repo: Path) -> None:
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _bootstrap_veeksha_python() -> None:
    veeksha_repo = _find_veeksha_repo()
    if veeksha_repo is None:
        return

    candidate = veeksha_repo / ".venv" / "bin" / "python"
    if candidate.exists() and Path(sys.executable).resolve() != candidate.resolve():
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{veeksha_repo}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(veeksha_repo)
        )
        os.execve(
            str(candidate),
            [str(candidate), "-m", MODULE, *sys.argv[1:]],
            env,
        )

    _prepend_pythonpath(veeksha_repo)


_bootstrap_veeksha_python()

from veeksha.case_studies.workload_deployment_claim_search import main


if __name__ == "__main__":
    raise SystemExit(main())
