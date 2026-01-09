"""Config schema introspection and documentation generation.

This module provides utilities to introspect the veeksha config system and generate
documentation in various formats (Markdown, YAML schema, JSON Schema).

The config system uses nested frozen dataclasses with polymorphic types (BasePolyConfig).
This module walks the dataclass tree and extracts:
- Field names, types, defaults, and help text
- Polymorphic type variants
- Nested structure

Example usage:
    >>> from veeksha.config.schema import ConfigSchema
    >>> from veeksha.config.benchmark import BenchmarkConfig
    >>> schema = ConfigSchema.from_dataclass(BenchmarkConfig)
    >>> print(schema.to_yaml())
    >>> print(schema.to_markdown())
"""

from __future__ import annotations

import json
from dataclasses import MISSING, Field, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type, Union, get_args, get_origin

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.utils import get_all_subclasses, is_optional, to_snake_case


class FieldSchema:
    """Schema representation of a single field."""

    def __init__(
        self,
        name: str,
        field_type: str,
        default: Any = MISSING,
        help_text: Optional[str] = None,
        required: bool = False,
        children: Optional[Dict[str, "FieldSchema"]] = None,
        variants: Optional[Dict[str, Dict[str, "FieldSchema"]]] = None,
        is_polymorphic: bool = False,
        is_list: bool = False,
    ):
        self.name = name
        self.field_type = field_type
        self.default = default
        self.help_text = help_text
        self.required = required
        self.children = children or {}
        self.variants = variants or {}
        self.is_polymorphic = is_polymorphic
        self.is_list = is_list

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "name": self.name,
            "type": self.field_type,
            "required": self.required,
        }
        if self.help_text:
            result["description"] = self.help_text
        if self.default is not MISSING:
            result["default"] = self._serialize_default(self.default)
        if self.children:
            result["properties"] = {
                name: child.to_dict() for name, child in self.children.items()
            }
        if self.variants:
            result["variants"] = {
                variant_name: {
                    name: child.to_dict() for name, child in variant_fields.items()
                }
                for variant_name, variant_fields in self.variants.items()
            }
        if self.is_list:
            result["is_list"] = True
        return result

    def _serialize_default(self, value: Any) -> Any:
        """Serialize default value to JSON-compatible format."""
        if value is MISSING:
            return None
        if callable(value):
            try:
                value = value()
            except Exception:
                return "<callable>"
        if isinstance(value, Enum):
            return value.name.lower()
        if is_dataclass(value):
            return f"<{value.__class__.__name__}>"
        return value


class ConfigSchema:
    """Schema representation of a config dataclass."""

    def __init__(self, root: FieldSchema, source_class: Type):
        self.root = root
        self.source_class = source_class

    @classmethod
    def from_dataclass(
        cls, dataclass_type: Type, name: Optional[str] = None
    ) -> "ConfigSchema":
        """Build schema from a dataclass type."""
        root = cls._extract_field_schema(
            dataclass_type, name=name or to_snake_case(dataclass_type.__name__)
        )
        return cls(root=root, source_class=dataclass_type)

    @classmethod
    def _extract_field_schema(
        cls, field_type: Type, name: str, field_obj: Field | None = None
    ) -> FieldSchema:
        """Recursively extract schema from a type."""
        help_text = None
        default = MISSING
        required = True

        # Field metadata / defaults
        if field_obj is not None:
            help_text = field_obj.metadata.get("help") if field_obj.metadata else None
            if field_obj.default is not MISSING:
                default = field_obj.default
                required = False
            if field_obj.default_factory is not MISSING:  # type: ignore[attr-defined]
                default = field_obj.default_factory  # type: ignore[attr-defined]
                required = False

        # Handle Optional[T]
        actual_type = field_type
        if is_optional(field_type):
            required = False
            actual_type = next(
                t for t in get_args(field_type) if t is not type(None)
            )  # noqa: E721

        # Handle List[T]
        is_list_type = get_origin(actual_type) is list
        if is_list_type:
            item_type = get_args(actual_type)[0]
            child_schema = cls._extract_field_schema(item_type, name=name)
            # Wrap the child schema to mark list
            return FieldSchema(
                name=name,
                field_type=f"List[{child_schema.field_type}]",
                default=default,
                help_text=help_text,
                required=required,
                children=child_schema.children,
                variants=child_schema.variants,
                is_polymorphic=child_schema.is_polymorphic,
                is_list=True,
            )

        # Handle polymorphic configs
        if (
            isinstance(actual_type, type)
            and issubclass(actual_type, BasePolyConfig)
            and actual_type is not BasePolyConfig
        ):
            return cls._extract_polymorphic_schema(
                base_type=actual_type,
                name=name,
                help_text=help_text,
                default=default,
                required=required,
            )

        # Handle nested dataclasses
        if is_dataclass(actual_type):
            children: Dict[str, FieldSchema] = {}
            for f in fields(actual_type):
                child = cls._extract_field_schema(f.type, f.name, f)
                children[f.name] = child
            return FieldSchema(
                name=name,
                field_type=cls._get_type_name(actual_type),
                default=default,
                help_text=help_text,
                required=required,
                children=children,
            )

        # Primitive / fallback
        type_name = cls._get_type_name(actual_type)
        return FieldSchema(
            name=name,
            field_type=type_name,
            default=default,
            help_text=help_text,
            required=required,
            is_list=is_list_type,
        )

    @classmethod
    def _extract_polymorphic_schema(
        cls,
        base_type: Type[BasePolyConfig],
        name: str,
        help_text: Optional[str],
        default: Any,
        required: bool,
        is_list: bool = False,
    ) -> FieldSchema:
        """Extract schema for a polymorphic config type."""
        variants: Dict[str, Dict[str, FieldSchema]] = {}
        subclasses = get_all_subclasses(base_type)

        # Fields common to the base class (if any)
        base_children: Dict[str, FieldSchema] = {}
        for f in fields(base_type):
            if f.name == "type":
                continue
            base_children[f.name] = cls._extract_field_schema(f.type, f.name, f)

        for subclass in subclasses:
            variant_fields: Dict[str, FieldSchema] = {}
            for f in fields(subclass):
                if f.name == "type":
                    continue
                variant_fields[f.name] = cls._extract_field_schema(f.type, f.name, f)
            variant_name = cls._get_variant_name(subclass)
            variants[variant_name] = variant_fields
        return FieldSchema(
            name=name,
            field_type="polymorphic",
            default=default,
            help_text=help_text,
            required=required,
            children=base_children,
            variants=variants,
            is_polymorphic=True,
            is_list=is_list,
        )

    @staticmethod
    def _normalize_variant_value(value: Any) -> str:
        if isinstance(value, Enum):
            return value.name.lower()
        return str(value)

    @classmethod
    def _get_variant_name(cls, poly_subclass: Type[BasePolyConfig]) -> str:
        # First, try calling get_type() classmethod (preferred for BasePolyConfig subclasses)
        try:
            type_value = poly_subclass.get_type()
            return cls._normalize_variant_value(type_value)
        except NotImplementedError:
            pass

        # Fallback: look for a 'type' field in the dataclass
        for f in fields(poly_subclass):
            if f.name != "type":
                continue
            if f.default is not MISSING:
                return cls._normalize_variant_value(f.default)
            if getattr(f, "default_factory", MISSING) is not MISSING:  # type: ignore[attr-defined]
                try:
                    return cls._normalize_variant_value(f.default_factory())  # type: ignore[attr-defined]
                except Exception:
                    pass
            type_hint = f.type
            if get_origin(type_hint) is Literal:
                literal_args = get_args(type_hint)
                if literal_args:
                    return cls._normalize_variant_value(literal_args[0])
        return to_snake_case(poly_subclass.__name__)

    @staticmethod
    def _get_type_name(field_type: Type) -> str:
        # typing types
        origin = get_origin(field_type)
        if origin is list:
            inner = get_args(field_type)[0]
            return f"List[{ConfigSchema._get_type_name(inner)}]"
        if origin is Union:
            args = [ConfigSchema._get_type_name(a) for a in get_args(field_type)]
            return " | ".join(args)

        # plain python types
        if isinstance(field_type, type):
            return field_type.__name__
        return str(field_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.root.to_dict()

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_yaml(self, include_help: bool = True) -> str:
        """Generate YAML schema representation with comments."""
        lines = []
        self._yaml_field(self.root, lines, indent=0, include_help=include_help)
        return "\n".join(lines)

    def _yaml_field(
        self,
        field: FieldSchema,
        lines: List[str],
        indent: int,
        include_help: bool,
        is_root: bool = True,
    ) -> None:
        prefix = "  " * indent

        if include_help and field.help_text:
            for line in self._wrap_text(field.help_text):
                lines.append(f"{prefix}# {line}")

        if field.is_polymorphic:
            self._yaml_polymorphic_field(field, lines, indent, include_help)
        elif field.children:
            lines.append(f"{prefix}{field.name}:")
            for child in field.children.values():
                self._yaml_field(child, lines, indent + 1, include_help, False)
        else:
            default_str = (
                f"  # default: {self._format_default(field.default)}"
                if field.default is not MISSING
                else ""
            )
            lines.append(f"{prefix}{field.name}: <{field.field_type}>{default_str}")

    def _yaml_polymorphic_field(
        self,
        field: FieldSchema,
        lines: List[str],
        indent: int,
        include_help: bool,
    ) -> None:
        prefix = "  " * indent
        variant_names = list(field.variants.keys())
        default_type = (
            field.default
            if field.default is not MISSING
            else variant_names[0] if variant_names else "unknown"
        )

        lines.append(f"{prefix}{field.name}:")
        if variant_names:
            lines.append(
                f"{prefix}  type: <{' | '.join(variant_names)}>  # default: {default_type}"
            )
        else:
            lines.append(f"{prefix}  type: <unknown>  # default: {default_type}")

        # Common fields
        for child in field.children.values():
            self._yaml_field(child, lines, indent + 1, include_help, False)

        # Variant-specific fields as comments
        for variant_name, variant_fields in field.variants.items():
            lines.append(f"{prefix}  # {variant_name}:")
            for child in variant_fields.values():
                rendered = []
                self._yaml_field(child, rendered, indent + 2, include_help, False)
                for line in rendered:
                    lines.append(f"{prefix}  # {line[len(prefix)+2:]}")

    def _format_default(self, value: Any) -> str:
        if value is MISSING:
            return ""
        if callable(value):
            return "<callable>"
        if isinstance(value, Enum):
            return value.name.lower()
        if is_dataclass(value):
            return f"<{value.__class__.__name__}>"
        if isinstance(value, str):
            return f"'{value}'"
        return str(value)

    @staticmethod
    def _wrap_text(text: str, max_width: int = 70) -> List[str]:
        words = text.split()
        lines: List[str] = []
        current = []
        width = 0
        for w in words:
            if width + len(w) + (1 if current else 0) > max_width:
                lines.append(" ".join(current))
                current = [w]
                width = len(w)
            else:
                current.append(w)
                width += len(w) + (1 if current else 0)
        if current:
            lines.append(" ".join(current))
        return lines

    def to_markdown(self, title: Optional[str] = None) -> str:
        """Generate Markdown documentation."""
        lines = []
        if title:
            lines.append(f"# {title}")
            lines.append("")

        self._markdown_field(self.root, lines, depth=0, path="")
        return "\n".join(lines)

    def _markdown_field(
        self,
        field: FieldSchema,
        lines: List[str],
        depth: int,
        path: str,
    ) -> None:
        current_path = f"{path}.{field.name}" if path else field.name

        if field.is_polymorphic:
            self._markdown_polymorphic_field(field, lines, depth, path)
            return
        if field.children:
            heading_level = min(depth + 2, 6)
            lines.append(f"{'#' * heading_level} `{field.name}`")
            lines.append("")
            lines.append(f"**Path:** `{current_path}`")
            lines.append(f"**Type:** Object")
            if field.default is not MISSING:
                lines.append(f"**Default:** `{self._format_default(field.default)}`")
            if field.help_text:
                lines.append("")
                lines.append(field.help_text)
            lines.append("")
            for child in field.children.values():
                self._markdown_field(child, lines, depth + 1, current_path)
        else:
            heading_level = min(depth + 2, 6)
            lines.append(f"{'#' * heading_level} `{field.name}`")
            lines.append("")
            lines.append(f"**Path:** `{current_path}`")
            lines.append(f"**Type:** `{field.field_type}`")
            if field.default is not MISSING:
                lines.append(f"**Default:** `{self._format_default(field.default)}`")
            lines.append(f"**Required:** `{field.required}`")
            if field.help_text:
                lines.append("")
                lines.append(field.help_text)
            lines.append("")

    def _markdown_polymorphic_field(
        self,
        field: FieldSchema,
        lines: List[str],
        depth: int,
        path: str,
    ) -> None:
        heading_level = min(depth + 2, 6)
        lines.append(f"{'#' * heading_level} `{field.name}`")
        lines.append("")
        if field.help_text:
            lines.append(field.help_text)
            lines.append("")
        variant_names = list(field.variants.keys())
        default_type = (
            field.default
            if field.default is not MISSING
            else variant_names[0] if variant_names else "unknown"
        )
        lines.append(f"**Type:** Polymorphic (`{' | '.join(variant_names)}`)")
        lines.append(f"**Default type:** `{default_type}`")
        lines.append("")

        if field.children:
            lines.append("**Common options:**")
            lines.append("")
            for child in field.children.values():
                self._markdown_field(child, lines, depth + 1, path)

        for variant_name, variant_fields in field.variants.items():
            lines.append(f"**Variant `{variant_name}`:**")
            lines.append("")
            for child in variant_fields.values():
                self._markdown_field(child, lines, depth + 1, path)

    def to_json_schema(self) -> Dict[str, Any]:
        """Generate JSON Schema for YAML validation and IDE autocompletion."""
        return self._json_schema_for_field(self.root)

    def _json_schema_for_field(self, field: FieldSchema) -> Dict[str, Any]:
        """Recursively generate JSON Schema for a field."""
        schema: Dict[str, Any] = {}

        if field.help_text:
            schema["description"] = field.help_text

        # Lists - check this FIRST since a field can be both is_list and is_polymorphic
        if field.is_list:
            schema["type"] = "array"
            if field.is_polymorphic and field.variants:
                # list of polymorphic items
                item_poly: Dict[str, Any] = {"oneOf": []}
                for variant_name, variant_fields in field.variants.items():
                    variant_schema = {
                        "type": "object",
                        "properties": {"type": {"const": variant_name}},
                        "required": ["type"],
                        "additionalProperties": False,
                    }
                    for child in field.children.values():
                        variant_schema["properties"][child.name] = (
                            self._json_schema_for_field(child)
                        )
                        if child.required:
                            variant_schema.setdefault("required", []).append(child.name)
                    for child in variant_fields.values():
                        variant_schema["properties"][child.name] = (
                            self._json_schema_for_field(child)
                        )
                        if child.required:
                            variant_schema.setdefault("required", []).append(child.name)
                    item_poly["oneOf"].append(variant_schema)
                if item_poly["oneOf"]:
                    schema["items"] = item_poly
            elif field.children:
                # list of dataclass items (non-polymorphic)
                item_obj = {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                }
                req = []
                for child in field.children.values():
                    item_obj["properties"][child.name] = self._json_schema_for_field(
                        child
                    )
                    if child.required:
                        req.append(child.name)
                if req:
                    item_obj["required"] = req
                schema["items"] = item_obj
            else:
                # primitive list
                schema["items"] = {
                    "type": self._python_type_to_json_schema_type(field.field_type)
                }
            if field.default is not MISSING:
                schema["default"] = self._serialize_default_for_json(field.default)
            return schema

        # Polymorphic (non-list)
        if field.is_polymorphic:
            if field.variants:  # only emit oneOf when variants exist
                one_of = []
                for variant_name, variant_fields in field.variants.items():
                    variant_schema = {
                        "type": "object",
                        "properties": {
                            "type": {"const": variant_name},
                        },
                        "required": ["type"],
                        "additionalProperties": False,
                    }
                    # common/base fields
                    for child in field.children.values():
                        variant_schema["properties"][child.name] = (
                            self._json_schema_for_field(child)
                        )
                        if child.required:
                            variant_schema.setdefault("required", []).append(child.name)
                    # variant-specific fields
                    for child in variant_fields.values():
                        variant_schema["properties"][child.name] = (
                            self._json_schema_for_field(child)
                        )
                        if child.required:
                            variant_schema.setdefault("required", []).append(child.name)
                    one_of.append(variant_schema)
                schema["oneOf"] = one_of
            else:
                # fallback to object with common children (no variants discovered)
                schema["type"] = "object"
                schema["properties"] = {}
                for child in field.children.values():
                    schema["properties"][child.name] = self._json_schema_for_field(
                        child
                    )
                schema["additionalProperties"] = False

            if field.default is not MISSING:
                schema["default"] = self._serialize_default_for_json(field.default)
            return schema

        # Objects (dataclasses)
        if field.children:
            schema["type"] = "object"
            schema["properties"] = {}
            required_fields = []
            for child in field.children.values():
                schema["properties"][child.name] = self._json_schema_for_field(child)
                if child.required:
                    required_fields.append(child.name)
            if required_fields:
                schema["required"] = required_fields
            schema["additionalProperties"] = False
            if field.default is not MISSING:
                schema["default"] = self._serialize_default_for_json(field.default)
            return schema

        # Primitives
        base_type = self._python_type_to_json_schema_type(field.field_type)
        base_schema = {"type": base_type}

        # Support !expand by allowing a list of the primitive
        # We essentially allow T | List[T]
        expand_schema = {"type": "array", "items": {"type": base_type}}

        # Combine into oneOf
        schema["oneOf"] = [base_schema, expand_schema]

        if field.default is not MISSING:
            # Default applies to the whole field (usually the primitive value)
            schema["default"] = self._serialize_default_for_json(field.default)
            # Also attach default to the base primitive option for clarity/tools
            base_schema["default"] = self._serialize_default_for_json(field.default)

        return schema

    @staticmethod
    def _python_type_to_json_schema_type(type_name: str) -> str:
        mapping = {
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "str": "string",
            "List": "array",
            "list": "array",
            "dict": "object",
            "Dict": "object",
        }
        return mapping.get(type_name, "string")

    def _serialize_default_for_json(self, value: Any) -> Any:
        """Serialize default value for JSON Schema."""
        if value is MISSING:
            return None
        if callable(value):
            try:
                value = value()
            except Exception:
                return None
        return self._serialize_default_value(value)

    def _serialize_default_value(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.name.lower()
        if is_dataclass(value):
            return f"<{value.__class__.__name__}>"
        if isinstance(value, dict):
            return {k: self._serialize_default_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_default_value(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)


def get_benchmark_schema() -> ConfigSchema:
    """Get the schema for BenchmarkConfig."""
    from veeksha.config.benchmark import BenchmarkConfig

    return ConfigSchema.from_dataclass(BenchmarkConfig)


def get_capacity_search_schema() -> ConfigSchema:
    """Get the schema for CapacitySearchConfig."""
    from veeksha.config.capacity_search import CapacitySearchConfig

    return ConfigSchema.from_dataclass(CapacitySearchConfig)
