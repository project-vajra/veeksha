"""Generate Sphinx documentation for veeksha configuration.

This module generates RST files for the Sphinx documentation from the config schema.
Each dataclass gets its own RST file with a readable pseudo-code format.
"""

from __future__ import annotations

import re
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, get_args, get_origin

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.utils import get_all_subclasses, is_optional


def _to_human_readable(class_name: str) -> str:
    """Convert a class name to human-readable format.

    E.g., 'AudioChannelGeneratorConfig' → 'Audio Channel Generator'
         'OpenAIChatCompletionsClientConfig' → 'OpenAI Chat Completions Client'
         'RAGTraceFlavorConfig' → 'RAG Trace Flavor'
    """
    # Remove 'Config' suffix
    name = class_name
    if name.endswith("Config"):
        name = name[:-6]

    # Use regex to insert spaces at camelCase boundaries
    # This splits before a capital that's followed by lowercase OR
    # before the last capital in a sequence of capitals
    result = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)

    # Fix known brand names that may have been incorrectly split
    fixes = {
        "Open AI": "OpenAI",
        "LM Eval": "LMEval",
        "Lm eval": "LMEval",
        # These don't get split by the regex since they have no internal capitals
        "Vllm": "vLLM",
        "Sglang": "SGLang",
    }
    for wrong, correct in fixes.items():
        result = result.replace(wrong, correct)

    return result


def generate_sphinx_docs(output_dir: str = "docs/config_reference") -> None:
    """Generate Sphinx RST documentation for all config types."""
    from veeksha.config.benchmark import BenchmarkConfig
    from veeksha.config.capacity_search import CapacitySearchConfig

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create api_reference subdirectory
    api_path = output_path / "api_reference"
    api_path.mkdir(parents=True, exist_ok=True)

    # Collect all dataclasses from root configs
    all_classes: Dict[str, Type] = {}
    _collect_dataclasses(BenchmarkConfig, all_classes)
    _collect_dataclasses(CapacitySearchConfig, all_classes)

    # Generate RST for each class
    for class_name, cls in sorted(all_classes.items()):
        rst_content = _generate_class_rst(cls, all_classes)
        (api_path / f"{class_name}.rst").write_text(rst_content)

    # Generate API reference index
    api_index = _generate_api_index(all_classes)
    (api_path / "index.rst").write_text(api_index)

    # Generate main config reference index
    main_index = _generate_main_index()
    (output_path / "index.rst").write_text(main_index)

    print(f"Generated API reference for {len(all_classes)} classes in {api_path}")


def _collect_dataclasses(cls: Type, collected: Dict[str, Type]) -> None:
    """Recursively collect all dataclass types from a root class."""
    if not is_dataclass(cls):
        return

    class_name = cls.__name__
    if class_name in collected:
        return

    collected[class_name] = cls

    # Collect from fields
    for f in fields(cls):
        field_type = f.type
        _collect_from_type(field_type, collected)


def _collect_from_type(field_type: Type, collected: Dict[str, Type]) -> None:
    """Extract dataclass types from a type annotation."""
    # Handle Optional[T]
    if is_optional(field_type):
        args = get_args(field_type)
        for arg in args:
            if arg is not type(None):
                _collect_from_type(arg, collected)
        return

    # Handle List[T]
    origin = get_origin(field_type)
    if origin is list:
        args = get_args(field_type)
        if args:
            _collect_from_type(args[0], collected)
        return

    # Handle polymorphic base classes
    if isinstance(field_type, type) and issubclass(field_type, BasePolyConfig):
        _collect_dataclasses(field_type, collected)
        # Also collect all subclasses
        for subclass in get_all_subclasses(field_type):
            _collect_dataclasses(subclass, collected)
        return

    # Handle regular dataclasses
    if is_dataclass(field_type):
        _collect_dataclasses(field_type, collected)


def _generate_class_rst(cls: Type, all_classes: Dict[str, Type]) -> str:
    """Generate RST documentation for a single dataclass."""
    lines: List[str] = []
    class_name = cls.__name__

    # Title
    lines.append(class_name)
    lines.append("=" * len(class_name))
    lines.append("")

    # Class signature
    parent_classes = [
        base.__name__
        for base in cls.__bases__
        if base.__name__ != "object" and is_dataclass(base)
    ]

    if parent_classes:
        parent_links = ", ".join(
            f":doc:`{p}`" if p in all_classes else f"``{p}``" for p in parent_classes
        )
        lines.append(f"**class** ``{class_name}`` **(** {parent_links} **)**")
    else:
        lines.append(f"**class** ``{class_name}``")
    lines.append("")

    # Docstring - skip auto-generated constructor signatures
    if cls.__doc__:
        docstring = cls.__doc__.strip()
        # Skip if it looks like an auto-generated dataclass signature
        if not (docstring.startswith(class_name + "(") and "=" in docstring):
            # Extract only first paragraph (stop at Attributes:, Args:, etc.)
            first_para = _extract_first_paragraph(docstring)
            if first_para:
                lines.append(f"    {first_para}")
                lines.append("")

    # Polymorphic type info (if this is a poly subclass)
    if (
        isinstance(cls, type)
        and issubclass(cls, BasePolyConfig)
        and cls is not BasePolyConfig
    ):
        has_header = False

        # 1. Try to display the specific type of this class
        try:
            type_value = cls.get_type()
            if hasattr(type_value, "name"):
                type_value = type_value.name.lower()
            lines.append("**Polymorphic Type:**")
            lines.append("")
            lines.append(f"    ``type: {type_value}``")
            has_header = True
        except NotImplementedError:
            pass

        # 2. Display all available types for this family
        # Find the root polymorphic base (closest to BasePolyConfig)
        mro = cls.mro()
        # Filter for custom classes that inherit from BasePolyConfig
        poly_parents = [
            c
            for c in mro
            if issubclass(c, BasePolyConfig)
            and c is not BasePolyConfig
            and c is not object
        ]

        variant_lines = []
        if poly_parents:
            poly_root = poly_parents[-1]
            variant_lines = _get_variant_lines(poly_root, all_classes)

            if variant_lines:
                if not has_header:
                    lines.append("**Polymorphic Type:**")
                    lines.append("")
                else:
                    lines.append("")

                lines.append(f"    All ``{poly_root.__name__}`` types:")
                lines.append("")
                for v_line in variant_lines:
                    lines.append(f"    {v_line}")

        # Add trailing newline if we added any polymorphic info
        if has_header or (poly_parents and variant_lines):
            lines.append("")

    # Fields section
    class_fields = list(fields(cls))
    if class_fields:
        lines.append("**Fields:**")
        lines.append("")

        for f in class_fields:
            if f.name == "type":  # Skip polymorphic type discriminator
                continue
            _generate_field_rst(f, lines, all_classes)

    return "\n".join(lines)


def _generate_field_rst(f: Any, lines: List[str], all_classes: Dict[str, Type]) -> None:
    """Generate RST for a single field."""
    field_name = f.name
    field_type = f.type
    help_text = f.metadata.get("help", "") if f.metadata else ""

    # Get type string and check for polymorphic
    type_str, is_poly, poly_base = _get_type_info(field_type, all_classes)

    # Get default value
    default_str = _get_default_str(f)

    # Field line
    lines.append(f"``{field_name}`` : {type_str} = {default_str}")

    # Help text
    if help_text:
        # Clean up help text (remove "Available: ..." since we show it below)
        clean_help = help_text.split(". Available:")[0].strip()
        if clean_help:
            lines.append(f"    {clean_help}")

    # Available types for polymorphic fields
    if is_poly and poly_base:
        variant_lines = _get_variant_lines(poly_base, all_classes)
        if variant_lines:
            lines.append("")
            lines.append("    Available types:")
            lines.append("")
            for v_line in variant_lines:
                lines.append(f"    {v_line}")

    lines.append("")


def _get_type_info(
    field_type: Type, all_classes: Dict[str, Type]
) -> tuple[str, bool, Optional[Type]]:
    """Get type string, whether it's polymorphic, and the polymorphic base if so."""
    actual_type = field_type
    is_optional_type = False
    is_list_type = False

    # Handle Optional[T]
    if is_optional(field_type):
        is_optional_type = True
        args = get_args(field_type)
        actual_type = next((t for t in args if t is not type(None)), field_type)

    # Handle List[T]
    origin = get_origin(actual_type)
    if origin is list:
        is_list_type = True
        args = get_args(actual_type)
        if args:
            actual_type = args[0]

    # Check if polymorphic
    is_poly = False
    poly_base = None
    if isinstance(actual_type, type) and issubclass(actual_type, BasePolyConfig):
        is_poly = True
        poly_base = actual_type

    # Build type string
    if isinstance(actual_type, type):
        type_name = actual_type.__name__
        if type_name in all_classes:
            type_str = f":doc:`{type_name}`"
        else:
            type_str = f"*{type_name}*"
    else:
        type_str = f"*{actual_type}*"

    if is_list_type:
        type_str = f"*list* [ {type_str} ]"
    if is_optional_type:
        type_str = f"*Optional* [ {type_str} ]"

    return type_str, is_poly, poly_base


def _get_default_str(f: Any) -> str:
    """Get default value string for a field."""
    if f.default is not MISSING:
        return _format_default(f.default)
    if f.default_factory is not MISSING:
        try:
            value = f.default_factory()
            return _format_default(value)
        except Exception:
            return "``<factory>``"
    return "*required*"


def _format_default(value: Any) -> str:
    """Format a default value for display."""
    if value is MISSING:
        return "*required*"
    if isinstance(value, type):
        return f":doc:`{value.__name__}`"
    if hasattr(value, "__dataclass_fields__"):
        return f":doc:`{value.__class__.__name__}`"
    if hasattr(value, "name"):  # Enum
        return f"``{value.name.lower()}``"
    if isinstance(value, str):
        return f'``"{value}"``'
    if isinstance(value, bool):
        return f"``{value}``"
    if isinstance(value, (int, float)):
        return f"``{value}``"
    if isinstance(value, list):
        if not value:
            return "``[]``"
        # For lists of dataclasses, show simplified form
        if value and hasattr(value[0], "__dataclass_fields__"):
            return f"``[<{value[0].__class__.__name__}>]``"
        # Keep short lists, simplify long ones
        str_repr = str(value)
        if len(str_repr) > 50:
            return f"``[...]``"
        return f"``{value}``"
    # For any complex object, simplify
    str_repr = str(value)
    if len(str_repr) > 50:
        return f"``<{type(value).__name__}>``"
    return f"``{value}``"


def _extract_first_paragraph(docstring: str) -> str:
    """Extract only the first paragraph from a docstring.

    Stops at section headers like Attributes:, Args:, Returns:, etc.
    """
    lines = docstring.split("\n")
    result_lines = []

    for line in lines:
        stripped = line.strip()
        # Stop at section headers (commonly used in docstrings)
        if stripped and stripped.endswith(":") and stripped[:-1].isalpha():
            break
        # Stop at blank line after we have content
        if not stripped and result_lines:
            break
        if stripped:
            result_lines.append(stripped)

    return " ".join(result_lines)


def _get_variant_lines(
    poly_base: Type[BasePolyConfig], all_classes: Dict[str, Type]
) -> List[str]:
    """Get formatted list of polymorphic variants."""
    subclasses = get_all_subclasses(poly_base)
    lines = []
    for subclass in subclasses:
        class_name = subclass.__name__
        try:
            type_value = subclass.get_type()
            if hasattr(type_value, "name"):
                type_value = type_value.name.lower()
        except NotImplementedError:
            type_value = class_name.lower()

        if class_name in all_classes:
            link = f":doc:`{class_name}`"
        else:
            link = f"``{class_name}``"

        lines.append(f"- ``{type_value}``: {link}")

    return lines


def _generate_api_index(all_classes: Dict[str, Type]) -> str:
    """Generate the API reference index page with grouped classes."""
    # Import base classes for grouping
    from veeksha.config.client import BaseClientConfig
    from veeksha.config.evaluator import (
        BaseChannelPerformanceConfig,
        BaseEvaluatorConfig,
    )
    from veeksha.config.generator.channel import BaseChannelGeneratorConfig
    from veeksha.config.generator.interval import BaseIntervalGeneratorConfig
    from veeksha.config.generator.length import BaseLengthGeneratorConfig
    from veeksha.config.generator.session import (
        BaseSessionGeneratorConfig,
        BaseTraceFlavorConfig,
    )
    from veeksha.config.generator.session_graph import BaseSessionGraphGeneratorConfig
    from veeksha.config.server import BaseServerConfig
    from veeksha.config.slo import BaseSloConfig
    from veeksha.config.traffic import BaseTrafficConfig

    # Group definitions: (Section Title, Base Class or None for root, classes list)
    groups: Dict[str, List[str]] = {
        "Root configurations": [],
        "Session generators": [],
        "Traffic schedulers": [],
        "Clients": [],
        "Servers": [],
        "Evaluators": [],
        "Channel performance": [],
        "Length generators": [],
        "Interval generators": [],
        "Channel generators": [],
        "Session graph generators": [],
        "Trace flavors": [],
        "SLO configs": [],
        "Other": [],
    }

    # Mapping of base classes to group names
    base_to_group: List[tuple[Type, str]] = [
        (BaseSessionGeneratorConfig, "Session generators"),
        (BaseTrafficConfig, "Traffic schedulers"),
        (BaseClientConfig, "Clients"),
        (BaseServerConfig, "Servers"),
        (BaseEvaluatorConfig, "Evaluators"),
        (BaseChannelPerformanceConfig, "Channel performance"),
        (BaseLengthGeneratorConfig, "Length generators"),
        (BaseIntervalGeneratorConfig, "Interval generators"),
        (BaseChannelGeneratorConfig, "Channel generators"),
        (BaseSessionGraphGeneratorConfig, "Session graph generators"),
        (BaseTraceFlavorConfig, "Trace flavors"),
        (BaseSloConfig, "SLO configs"),
    ]

    # Root configs
    root_configs = {"BenchmarkConfig", "CapacitySearchConfig"}

    # Classify each class
    for class_name, cls in all_classes.items():
        if class_name in root_configs:
            groups["Root configurations"].append(class_name)
            continue

        # Check against base classes
        found = False
        for base_cls, group_name in base_to_group:
            if isinstance(cls, type) and issubclass(cls, base_cls):
                groups[group_name].append(class_name)
                found = True
                break

        if not found:
            groups["Other"].append(class_name)

    # Build the RST content
    lines = [
        "API Reference",
        "=============",
        "",
        "This section contains the API reference for all configuration classes, organized by category.",
        "",
    ]

    # Generate sections
    for group_name, class_names in groups.items():
        if not class_names:
            continue

        # Section header
        lines.append(group_name)
        lines.append("-" * len(group_name))
        lines.append("")

        # Root configs get special treatment with descriptions
        if group_name == "Root configurations":
            lines.append(
                "- :doc:`BenchmarkConfig` - Configuration for ``veeksha.benchmark`` runs"
            )
            lines.append(
                "- :doc:`CapacitySearchConfig` - Configuration for ``veeksha.capacity_search`` runs"
            )
            lines.append("")
            continue

        # Toctree with human-readable display names
        lines.append(".. toctree::")
        lines.append("   :maxdepth: 1")
        lines.append("")

        for class_name in sorted(class_names):
            display_name = _to_human_readable(class_name)
            lines.append(f"   {display_name} <{class_name}>")

        lines.append("")

    return "\n".join(lines)


def _generate_main_index() -> str:
    """Generate the main config reference index."""
    return """Configuration Reference
=======================

This section provides a comprehensive reference for all configuration options in Veeksha.
Configuration can be provided via YAML files or CLI arguments.

.. tip::
   
   Use the interactive config explorer for an easier experience::
   
       python -m veeksha.cli.config explore

   Or generate a YAML schema template::
   
       python -m veeksha.cli.config show --format yaml


Quick links
-----------

- :doc:`api_reference/BenchmarkConfig` - Configuration for standard benchmark runs
- :doc:`api_reference/CapacitySearchConfig` - Configuration for capacity search experiments
- :doc:`api_reference/index` - Full API reference for all config classes


Understanding the config system
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


IDE autocompletion
------------------

See :doc:`/basic_usage/configuration` for instructions on how to set up YAML autocompletion and validation in your IDE.


.. toctree::
   :maxdepth: 2
   :hidden:

   api_reference/index
"""


if __name__ == "__main__":
    generate_sphinx_docs()
