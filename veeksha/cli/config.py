"""CLI for exploring and documenting veeksha configuration options.

This module provides commands for:
- Showing the full config schema as YAML
- Exploring specific config paths interactively
- Generating documentation in various formats
- Exporting JSON schema for IDE autocompletion

Usage:
    python -m veeksha.cli.config show [--format yaml|json|markdown]
    python -m veeksha.cli.config describe <path>
    python -m veeksha.cli.config export-schema [--output schema.json]
    python -m veeksha.cli.config explore
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from veeksha.config.schema import (
    ConfigSchema,
    FieldSchema,
    get_benchmark_schema,
    get_capacity_search_schema,
)


def cmd_show(args: argparse.Namespace) -> None:
    """Show the full config schema."""
    schema = _get_schema(args.config_type)

    if args.format == "yaml":
        print(schema.to_yaml(include_help=not args.no_help))
    elif args.format == "json":
        print(schema.to_json(indent=2))
    elif args.format == "markdown":
        title = f"Veeksha {args.config_type.replace('_', ' ').title()} Configuration Reference"
        print(schema.to_markdown(title=title))
    else:
        print(f"Unknown format: {args.format}", file=sys.stderr)
        sys.exit(1)


def cmd_describe(args: argparse.Namespace) -> None:
    """Describe a specific config path."""
    schema = _get_schema(args.config_type)
    path = args.path

    field = _find_field_by_path(schema.root, path)
    if field is None:
        print(f"Config path '{path}' not found.", file=sys.stderr)
        print("\nAvailable top-level options:")
        _print_available_options(schema.root)
        sys.exit(1)

    _print_field_description(field, path)


def cmd_export_schema(args: argparse.Namespace) -> None:
    """Export JSON schema for IDE autocompletion."""
    schema = _get_schema(args.config_type)
    json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": f"Veeksha {args.config_type} Configuration",
        "description": f"Configuration schema for veeksha {args.config_type}",
        **schema.to_json_schema(),
    }

    output = json.dumps(json_schema, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"JSON schema written to {args.output}")
    else:
        print(output)


def cmd_explore(args: argparse.Namespace) -> None:
    """Interactive exploration of config options."""
    try:
        from veeksha.cli.config_explorer import run_explorer

        run_explorer(args.config_type)
    except ImportError as e:
        print(f"Interactive explorer requires textual: {e}", file=sys.stderr)
        print("Install with: pip install textual", file=sys.stderr)
        sys.exit(1)


def cmd_generate_docs(args: argparse.Namespace) -> None:
    """Generate documentation files for Sphinx."""
    from veeksha.cli.config_docs_generator import generate_sphinx_docs

    generate_sphinx_docs(args.output_dir)
    print(f"Documentation generated in {args.output_dir}")


def _get_schema(config_type: str) -> ConfigSchema:
    """Get schema for the specified config type."""
    if config_type == "benchmark":
        return get_benchmark_schema()
    elif config_type == "capacity_search":
        return get_capacity_search_schema()
    else:
        print(f"Unknown config type: {config_type}", file=sys.stderr)
        sys.exit(1)


def _find_field_by_path(root: FieldSchema, path: str) -> Optional[FieldSchema]:
    """Find a field by dot-separated path."""
    if not path:
        return root

    parts = path.split(".")
    current = root

    for part in parts:
        # Check in children
        if part in current.children:
            current = current.children[part]
            continue

        # Check in variant fields
        found = False
        for variant_name, variant_fields in current.variants.items():
            if part in variant_fields:
                current = variant_fields[part]
                found = True
                break
            # Also check if part is the variant type itself
            if part == variant_name:
                # Return a synthetic field for the variant
                return FieldSchema(
                    name=variant_name,
                    field_type="variant",
                    help_text=f"Variant type '{variant_name}' for {current.name}",
                    children=variant_fields,
                )

        if not found:
            return None

    return current


def _print_available_options(field: FieldSchema, prefix: str = "") -> None:
    """Print available options for a field."""
    for name, child in field.children.items():
        path = f"{prefix}.{name}" if prefix else name
        type_info = child.field_type
        if child.is_polymorphic:
            variants = list(child.variants.keys())
            type_info = f"({' | '.join(variants)})"
        print(f"  {path}: {type_info}")


def _print_field_description(field: FieldSchema, path: str) -> None:
    """Print detailed description of a field."""
    print(f"\n{'=' * 60}")
    print(f"  {path}")
    print(f"{'=' * 60}\n")

    if field.help_text:
        print(f"Description: {field.help_text}\n")

    print(f"Type: {field.field_type}")
    if field.is_list:
        print("(list)")
    if field.is_polymorphic:
        print("(polymorphic)")

    if field.default is not None:
        from dataclasses import MISSING

        if field.default is not MISSING:
            print(f"Default: {field.default}")

    if field.required:
        print("Required: Yes")
    else:
        print("Required: No")

    if field.is_polymorphic and field.variants:
        print(f"\nAvailable types: {', '.join(field.variants.keys())}")

        print("\n--- Variant Options ---")
        for variant_name, variant_fields in field.variants.items():
            print(f"\n  type: {variant_name}")
            if variant_fields:
                for vf_name, vf in variant_fields.items():
                    default_str = f" (default: {vf.default})" if vf.default else ""
                    print(f"    {vf_name}: {vf.field_type}{default_str}")
                    if vf.help_text:
                        # Indent help text
                        for line in _wrap_text(vf.help_text, 50):
                            print(f"      {line}")
            else:
                print("    (no additional options)")

    if field.children and not field.is_polymorphic:
        print("\n--- Nested Options ---")
        for child_name, child in field.children.items():
            default_str = f" (default: {child.default})" if child.default else ""
            poly_str = " [polymorphic]" if child.is_polymorphic else ""
            print(f"  {child_name}: {child.field_type}{poly_str}{default_str}")
            if child.help_text:
                for line in _wrap_text(child.help_text, 50):
                    print(f"    {line}")

    print()


def _wrap_text(text: str, max_width: int = 70) -> list[str]:
    """Wrap text to max width."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_width and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="veeksha config",
        description="Explore and document veeksha configuration options",
    )

    parser.add_argument(
        "--config-type",
        choices=["benchmark", "capacity_search"],
        default="benchmark",
        help="Which configuration to show (default: benchmark)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # show command
    show_parser = subparsers.add_parser(
        "show", help="Show the full configuration schema"
    )
    show_parser.add_argument(
        "--format",
        "-f",
        choices=["yaml", "json", "markdown"],
        default="yaml",
        help="Output format (default: yaml)",
    )
    show_parser.add_argument(
        "--no-help",
        action="store_true",
        help="Omit help text comments in YAML output",
    )
    show_parser.set_defaults(func=cmd_show)

    # describe command
    describe_parser = subparsers.add_parser(
        "describe", help="Describe a specific configuration option"
    )
    describe_parser.add_argument(
        "path",
        help="Dot-separated path to the config option (e.g., traffic_scheduler.type)",
    )
    describe_parser.set_defaults(func=cmd_describe)

    # export-schema command
    export_parser = subparsers.add_parser(
        "export-schema", help="Export JSON schema for IDE autocompletion"
    )
    export_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: stdout)",
    )
    export_parser.set_defaults(func=cmd_export_schema)

    # explore command
    explore_parser = subparsers.add_parser(
        "explore", help="Interactively explore configuration options"
    )
    explore_parser.set_defaults(func=cmd_explore)

    # generate-docs command
    docs_parser = subparsers.add_parser(
        "generate-docs", help="Generate Sphinx documentation for config reference"
    )
    docs_parser.add_argument(
        "--output-dir",
        "-o",
        default="docs/config_reference",
        help="Output directory for generated docs",
    )
    docs_parser.set_defaults(func=cmd_generate_docs)

    return parser


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        # Default to show with yaml format
        args.format = "yaml"
        args.no_help = False
        cmd_show(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
