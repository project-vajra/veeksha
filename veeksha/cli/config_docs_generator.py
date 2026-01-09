"""Generate Sphinx documentation for veeksha configuration.

This module generates RST files for the Sphinx documentation from the config schema.
"""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

from veeksha.config.schema import (
    ConfigSchema,
    FieldSchema,
    get_benchmark_schema,
    get_capacity_search_schema,
)


def generate_sphinx_docs(output_dir: str = "docs/config_reference") -> None:
    """Generate Sphinx RST documentation for all config types."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate index
    index_content = _generate_index()
    (output_path / "index.rst").write_text(index_content)

    # Generate benchmark config docs
    benchmark_schema = get_benchmark_schema()
    benchmark_content = _generate_config_docs(
        benchmark_schema,
        title="Benchmark Configuration",
        description="Configuration reference for ``veeksha.benchmark`` runs.",
    )
    (output_path / "benchmark.rst").write_text(benchmark_content)

    # Generate capacity search config docs
    capsearch_schema = get_capacity_search_schema()
    capsearch_content = _generate_config_docs(
        capsearch_schema,
        title="Capacity Search Configuration",
        description="Configuration reference for ``veeksha.capacity_search`` runs.",
    )
    (output_path / "capacity_search.rst").write_text(capsearch_content)

    print(f"Generated documentation in {output_path}")


def _generate_index() -> str:
    """Generate the index.rst for config reference."""
    return """Configuration Reference
=======================

This section provides a comprehensive reference for all configuration options in Veeksha.
Configuration can be provided via YAML files or CLI arguments.

.. tip::
   
   Use the interactive config explorer for an easier experience::
   
       python -m veeksha.cli.config explore

   Or generate a YAML schema template::
   
       python -m veeksha.cli.config show --format yaml


Quick Links
-----------

- :doc:`benchmark` - Configuration for standard benchmark runs
- :doc:`capacity_search` - Configuration for capacity search experiments


Understanding the Config System
-------------------------------

Veeksha uses a **polymorphic configuration system**. Many options have a ``type`` field
that determines which variant is used, each with its own set of options.

For example, the ``traffic_scheduler`` can be either ``rate`` or ``concurrent``::

    # Rate-based traffic
    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        rate: 10.0  # 10 requests per second

    # Concurrency-based traffic
    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: 8
      rampup_seconds: 10


IDE Autocompletion
------------------

Export a JSON schema for YAML autocompletion in your IDE::

    python -m veeksha.cli.config export-schema -o veeksha-schema.json

Then configure your IDE to use this schema for ``*.veeksha.yml`` files.

For VS Code, add to ``.vscode/settings.json``::

    {
        "yaml.schemas": {
            "./veeksha-schema.json": "*.veeksha.yml"
        }
    }


.. toctree::
   :maxdepth: 2
   :hidden:

   benchmark
   capacity_search
"""


def _generate_config_docs(
    schema: ConfigSchema,
    title: str,
    description: str,
) -> str:
    """Generate RST documentation for a config schema."""
    lines = []

    # Title
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append(description)
    lines.append("")

    # Quick example
    lines.append("Example Configuration")
    lines.append("-" * 21)
    lines.append("")
    lines.append(".. code-block:: yaml")
    lines.append("")
    for line in schema.to_yaml(include_help=False).split("\n")[:30]:
        lines.append(f"    {line}")
    lines.append("    # ... (see full schema below)")
    lines.append("")

    # Full reference
    lines.append("Full Reference")
    lines.append("-" * 14)
    lines.append("")

    # Generate docs for each top-level field
    for field_name, field in schema.root.children.items():
        _generate_field_docs(field, lines, depth=0, path=field_name)

    return "\n".join(lines)


def _generate_field_docs(
    field: FieldSchema,
    lines: list[str],
    depth: int,
    path: str,
) -> None:
    """Generate RST documentation for a field."""
    # Section header
    header_char = ["~", "^", '"', "'"][min(depth, 3)]
    lines.append(f"``{path}``")
    lines.append(header_char * (len(path) + 4))
    lines.append("")

    # Description
    if field.help_text:
        lines.append(field.help_text)
        lines.append("")

    # Type info
    type_str = field.field_type
    if field.is_list:
        type_str = f"list[{type_str}]"
    if field.is_polymorphic:
        type_str = f"{type_str} (polymorphic)"

    lines.append(f"**Type:** ``{type_str}``")
    lines.append("")

    if field.default is not MISSING:
        default_val = _format_default(field.default)
        lines.append(f"**Default:** ``{default_val}``")
        lines.append("")

    # Polymorphic types
    if field.is_polymorphic and field.variants:
        variant_names = list(field.variants.keys())
        lines.append(f"**Available types:** ``{'``, ``'.join(variant_names)}``")
        lines.append("")

        # Common options
        if field.children:
            lines.append("**Common options** (available for all types):")
            lines.append("")
            _generate_options_table(field.children, lines)
            lines.append("")

        # Variant-specific options
        lines.append("**Type-specific options:**")
        lines.append("")

        for variant_name, variant_fields in field.variants.items():
            lines.append(f"When ``type: {variant_name}``:")
            lines.append("")
            if variant_fields:
                _generate_options_table(variant_fields, lines)
            else:
                lines.append("*No additional options.*")
            lines.append("")

            # Recurse for complex variant fields
            for child_name, child in variant_fields.items():
                if child.children or (child.is_polymorphic and child.variants):
                    _generate_field_docs(
                        child, lines, depth + 1, f"{path}.{child_name}"
                    )

    # Nested options (non-polymorphic)
    elif field.children:
        lines.append("**Options:**")
        lines.append("")
        _generate_options_table(field.children, lines)
        lines.append("")

        # Recurse for complex nested fields
        for child_name, child in field.children.items():
            if child.children or child.is_polymorphic:
                _generate_field_docs(child, lines, depth + 1, f"{path}.{child_name}")


def _generate_options_table(
    fields: dict[str, FieldSchema],
    lines: list[str],
) -> None:
    """Generate an RST table for options."""
    lines.append(".. list-table::")
    lines.append("   :header-rows: 1")
    lines.append("   :widths: 20 15 15 50")
    lines.append("")
    lines.append("   * - Option")
    lines.append("     - Type")
    lines.append("     - Default")
    lines.append("     - Description")

    for name, field in fields.items():
        type_str = field.field_type
        if field.is_list:
            type_str = f"list[{type_str}]"
        if field.is_polymorphic:
            type_str = f"{type_str}*"

        default_str = (
            _format_default(field.default)
            if field.default is not MISSING
            else "*required*"
        )
        desc = field.help_text or ""

        lines.append(f"   * - ``{name}``")
        lines.append(f"     - ``{type_str}``")
        lines.append(f"     - {default_str}")
        lines.append(f"     - {desc}")

    lines.append("")


def _format_default(value) -> str:
    """Format a default value for display."""
    if value is MISSING:
        return "required"
    if callable(value):
        try:
            value = value()
        except Exception:
            return "<factory>"
    if hasattr(value, "name"):  # Enum
        return value.name.lower()
    if hasattr(value, "__dataclass_fields__"):
        return f"<{value.__class__.__name__}>"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


if __name__ == "__main__":
    generate_sphinx_docs()
