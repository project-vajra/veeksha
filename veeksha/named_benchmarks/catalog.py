"""Load strict named benchmark manifests from disk or the packaged catalog."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from veeksha.named_benchmarks.schema import Benchmark, BenchmarkSchemaError, _stable_id

_CATALOG_DIRECTORY = "catalog"
_YAML_SUFFIXES = (".yaml", ".yml")


class BenchmarkNotFoundError(FileNotFoundError):
    """Raised when neither a requested path nor catalog entry exists."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise BenchmarkSchemaError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_benchmark(reference: str | Path) -> Benchmark:
    """Load and validate a benchmark by YAML path or packaged catalog ID.

    A value ending in ``.yml``/``.yaml`` or containing a path separator is
    always treated as a filesystem path. Other strings are stable catalog IDs.
    """

    if isinstance(reference, Path):
        return _load_path(reference)

    candidate = Path(reference).expanduser()
    if (
        candidate.exists()
        or candidate.suffix.lower() in _YAML_SUFFIXES
        or _looks_like_path(reference)
    ):
        return _load_path(candidate)
    return _load_catalog_id(reference)


def available_benchmarks() -> tuple[str, ...]:
    """Return stable IDs for YAML manifests in the packaged catalog."""

    catalog = resources.files("veeksha.named_benchmarks").joinpath(_CATALOG_DIRECTORY)
    if not catalog.is_dir():
        return ()
    ids = {
        entry.name[: -len(suffix)]
        for entry in catalog.iterdir()
        for suffix in _YAML_SUFFIXES
        if entry.is_file() and entry.name.endswith(suffix)
    }
    return tuple(sorted(ids))


def _load_path(path: Path) -> Benchmark:
    if not path.is_file():
        raise BenchmarkNotFoundError(f"benchmark manifest does not exist: {path}")
    return _parse_yaml(path.read_text(encoding="utf-8"), source=str(path))


def _load_catalog_id(catalog_id: str) -> Benchmark:
    stable_id = _stable_id(catalog_id, context="catalog id")
    catalog = resources.files("veeksha.named_benchmarks").joinpath(_CATALOG_DIRECTORY)
    for suffix in _YAML_SUFFIXES:
        resource = catalog.joinpath(f"{stable_id}{suffix}")
        if resource.is_file():
            benchmark = _parse_yaml(
                resource.read_text(encoding="utf-8"), source=f"catalog:{stable_id}"
            )
            if benchmark.id != stable_id:
                raise BenchmarkSchemaError(
                    f"catalog entry {stable_id!r} declares id {benchmark.id!r}"
                )
            return benchmark
    raise BenchmarkNotFoundError(
        f"unknown benchmark {stable_id!r}; available: "
        + (", ".join(available_benchmarks()) or "<none>")
    )


def _parse_yaml(text: str, *, source: str) -> Benchmark:
    try:
        document = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except BenchmarkSchemaError:
        raise
    except yaml.YAMLError as exc:
        raise BenchmarkSchemaError(f"invalid YAML in {source}: {exc}") from exc
    if document is None:
        raise BenchmarkSchemaError(f"benchmark manifest is empty: {source}")
    try:
        return Benchmark.from_mapping(document)
    except BenchmarkSchemaError as exc:
        raise BenchmarkSchemaError(
            f"invalid benchmark manifest {source}: {exc}"
        ) from exc


def _looks_like_path(value: str) -> bool:
    return "/" in value or "\\" in value or value.startswith((".", "~"))
