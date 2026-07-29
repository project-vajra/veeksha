"""Veeksha CLI parsing with correct optional nested-config semantics."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Iterator, TypeVar, cast

from vidhi import BaseCommand
from vidhi.cli import _is_base_command, _resolve_subcommand
from vidhi.nested_config import (
    ConfigWalker,
    _deep_merge,
    _explode_config,
    _extract_config_file_arg,
    _filter_non_selected_variant_defaults,
    _flat_to_nested,
    _handle_builtin_flags,
    _process_iterable_args,
    _validate_variant_fields,
)
from vidhi.utils import create_class_from_dict, load_yaml_config

ConfigT = TypeVar("ConfigT")


def parse_cli_sweep(
    config_class: type[ConfigT],
    *,
    args: list[str] | None = None,
    description: str | None = None,
) -> list[ConfigT]:
    """Parse one command or config class into all expanded configurations.

    Vidhi 0.0.9 records optional nested groups but still emits every child
    default when the group is absent. That turns ``endpoint: None`` into an
    invalid, empty ``EndpointConfig``. Veeksha removes only the unprovided
    defaults below those groups before object construction.
    """
    cli_args = list(args if args is not None else sys.argv[1:])
    if _is_base_command(config_class):
        base_command_class = cast(type[BaseCommand], config_class)
        subcommand_class, remaining_args = _resolve_subcommand(
            base_command_class,
            args=cli_args,
            description=description,
        )
        return cast(
            list[ConfigT],
            parse_cli_sweep(
                subcommand_class,
                args=remaining_args,
                description=description,
            ),
        )

    with _use_cli_args(cli_args):
        return _parse_config_sweep(config_class)


@contextmanager
def _use_cli_args(cli_args: list[str]) -> Iterator[None]:
    saved_argv = sys.argv
    program = saved_argv[0] if saved_argv else "veeksha"
    try:
        sys.argv = [program, *cli_args]
        yield
    finally:
        sys.argv = saved_argv


def _parse_config_sweep(config_class: type[ConfigT]) -> list[ConfigT]:
    walker = ConfigWalker(config_class)
    parser = walker.build_parser()

    _handle_builtin_flags(config_class, walker)
    config_file = _extract_config_file_arg(sys.argv)
    parsed_args, provided_arg_names = parser.parse_args()

    cli_provided_args = {
        key: value for key, value in parsed_args.items() if key in provided_arg_names
    }
    parsed_args = _materialize_collection_defaults(parsed_args)
    cli_provided_args = _materialize_collection_defaults(cli_provided_args)
    parsed_args = _process_iterable_args(
        parsed_args,
        walker.list_fields,
        walker.container_fields,
    )
    cli_provided_args = _process_iterable_args(
        cli_provided_args,
        walker.list_fields,
        walker.container_fields,
    )

    if not config_file:
        _validate_variant_fields(walker, [parsed_args], cli_provided_args)

    parsed_args = _filter_non_selected_variant_defaults(
        parsed_args,
        provided_arg_names,
        walker,
    )
    parsed_args = _drop_unprovided_optional_group_defaults(
        parsed_args,
        provided_arg_names,
        walker.args_with_default_none,
    )

    cli_nested = _flat_to_nested(parsed_args)
    cli_provided_nested = _flat_to_nested(cli_provided_args)

    loaded_configs: list[dict[str, Any]] = [{}]
    if config_file:
        loaded_configs = _explode_config(walker, load_yaml_config(config_file))

    configs: list[ConfigT] = []
    provided_keys = frozenset(cli_provided_args.keys())
    for file_config in loaded_configs:
        file_config = _process_iterable_args(
            file_config,
            walker.list_fields,
            walker.container_fields,
        )
        merged_config = _deep_merge(cli_nested, file_config)
        merged_config = _deep_merge(merged_config, cli_provided_nested)
        instance = create_class_from_dict(config_class, merged_config)
        # Stash which flat CLI keys the user actually set so named-benchmark
        # runs can reject overrides of frozen definition fields.
        try:
            object.__setattr__(instance, "_cli_provided_keys", provided_keys)
        except (AttributeError, TypeError):
            pass
        configs.append(instance)

    return configs


def _materialize_collection_defaults(
    flat_config: dict[str, Any],
) -> dict[str, Any]:
    collection_factories = (list, dict, set, tuple)
    return {
        key: value() if value in collection_factories else value
        for key, value in flat_config.items()
    }


def _drop_unprovided_optional_group_defaults(
    flat_config: dict[str, Any],
    provided_arg_names: set[str],
    default_none_paths: set[str],
) -> dict[str, Any]:
    optional_groups = {
        path
        for path in default_none_paths
        if any(key.startswith(f"{path}.") for key in flat_config)
    }
    return {
        key: value
        for key, value in flat_config.items()
        if key in provided_arg_names
        or not any(key.startswith(f"{group}.") for group in optional_groups)
    }
