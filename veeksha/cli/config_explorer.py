"""Interactive TUI explorer for veeksha configuration.

This module provides a terminal-based UI for exploring the veeksha config schema
using the textual library.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import (
    Footer,
    Header,
    Static,
    Tree,
)
from textual.widgets.tree import TreeNode

from veeksha.config.schema import (
    ConfigSchema,
    FieldSchema,
    get_benchmark_schema,
    get_capacity_search_schema,
)


class ConfigTree(Tree):
    """Tree widget for displaying config schema."""

    def __init__(self, schema: ConfigSchema, **kwargs):
        super().__init__(schema.root.name, **kwargs)
        self.schema = schema
        self._build_tree()

    def _build_tree(self) -> None:
        """Build the tree from the schema."""
        self._add_field_to_tree(self.root, self.schema.root)
        self.root.expand()

    def _add_field_to_tree(self, node: TreeNode, field: FieldSchema) -> None:
        """Recursively add a field to the tree."""
        node.data = field

        if field.is_polymorphic:
            # Add type selector
            type_node = node.add(
                f"type: {field.default or 'select...'}",
                data={"is_type_selector": True, "parent_field": field},
            )

            # Add common children
            for child_name, child in field.children.items():
                child_node = node.add(self._get_node_label(child), data=child)
                self._add_field_to_tree(child_node, child)

            # Add variant-specific fields as collapsed sections
            for variant_name, variant_fields in field.variants.items():
                if variant_fields:
                    variant_node = node.add(
                        f"[dim]when type={variant_name}:[/dim]",
                        data={"is_variant_header": True, "variant": variant_name},
                    )
                    for vf_name, vf in variant_fields.items():
                        vf_node = variant_node.add(self._get_node_label(vf), data=vf)
                        self._add_field_to_tree(vf_node, vf)

        elif field.children:
            for child_name, child in field.children.items():
                child_node = node.add(self._get_node_label(child), data=child)
                self._add_field_to_tree(child_node, child)

    def _get_node_label(self, field: FieldSchema) -> str:
        """Get the display label for a field node."""
        type_str = field.field_type
        if field.is_polymorphic:
            type_str = f"[cyan]{type_str}[/cyan]"
        elif field.children:
            type_str = f"[yellow]{type_str}[/yellow]"
        else:
            type_str = f"[green]{type_str}[/green]"

        if field.is_list:
            type_str = f"list[{type_str}]"

        required = "[red]*[/red]" if field.required else ""
        return f"{field.name}{required}: {type_str}"


class DetailPanel(Vertical):
    """Panel showing details of selected config option."""

    DEFAULT_CSS = """
    DetailPanel {
        width: 50%;
        height: 100%;
        border-left: solid $primary;
        padding: 1 2;
    }

    DetailPanel .title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    DetailPanel .field-info {
        margin-bottom: 1;
    }

    DetailPanel .help-text {
        color: $text-muted;
        margin-bottom: 1;
    }

    DetailPanel .section-header {
        text-style: bold;
        color: $secondary;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Select a config option to see details", id="detail-content")

    def update_detail(self, field: Optional[FieldSchema], path: str = "") -> None:
        """Update the detail panel with field information."""
        content = self.query_one("#detail-content", Static)

        if field is None:
            content.update("Select a config option to see details")
            return

        lines = []
        lines.append(f"[bold cyan]{path or field.name}[/bold cyan]")
        lines.append("")

        if field.help_text:
            lines.append(f"[dim]{field.help_text}[/dim]")
            lines.append("")

        lines.append(f"[bold]Type:[/bold] {field.field_type}")
        if field.is_list:
            lines.append("[bold]Container:[/bold] list")
        if field.is_polymorphic:
            lines.append("[bold]Polymorphic:[/bold] Yes")

        if field.default is not MISSING:
            default_val = field.default
            if callable(default_val):
                try:
                    default_val = default_val()
                    default_val = f"<{default_val.__class__.__name__}>"
                except Exception:
                    default_val = "<factory>"
            lines.append(f"[bold]Default:[/bold] {default_val}")

        lines.append(f"[bold]Required:[/bold] {'Yes' if field.required else 'No'}")

        if field.is_polymorphic and field.variants:
            lines.append("")
            lines.append("[bold yellow]Available Types:[/bold yellow]")
            for variant_name in field.variants.keys():
                lines.append(f"  • {variant_name}")

        if field.children and not field.is_polymorphic:
            lines.append("")
            lines.append("[bold yellow]Options:[/bold yellow]")
            for child_name, child in field.children.items():
                req = "[red]*[/red]" if child.required else ""
                lines.append(f"  • {child_name}{req}: {child.field_type}")

        content.update("\n".join(lines))


class ConfigExplorerApp(App):
    """Main TUI application for exploring config."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    #tree-container {
        width: 50%;
        height: 100%;
        border-right: solid $primary;
    }

    ConfigTree {
        height: 100%;
    }

    #search-box {
        dock: top;
        height: 3;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("e", "expand_all", "Expand All"),
        Binding("c", "collapse_all", "Collapse All"),
        Binding("/", "search", "Search"),
        Binding("y", "copy_yaml", "Copy YAML Path"),
    ]

    def __init__(self, config_type: str = "benchmark"):
        super().__init__()
        self.config_type = config_type
        if config_type == "benchmark":
            self.schema = get_benchmark_schema()
        else:
            self.schema = get_capacity_search_schema()

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="tree-container"):
            yield ConfigTree(self.schema, id="config-tree")
        yield DetailPanel()
        yield Footer()

    def on_mount(self) -> None:
        """Handle mount event."""
        self.title = f"Veeksha Config Explorer ({self.config_type})"
        self.sub_title = "Navigate with arrow keys, Enter to expand/collapse"

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        node = event.node
        data = node.data

        if data is None:
            return

        # Handle special node types
        if isinstance(data, dict):
            if data.get("is_type_selector"):
                # Show variant options
                parent_field = data["parent_field"]
                detail = self.query_one(DetailPanel)
                path = self._get_node_path(node)
                detail.update_detail(parent_field, path)
                return
            elif data.get("is_variant_header"):
                return

        if isinstance(data, FieldSchema):
            detail = self.query_one(DetailPanel)
            path = self._get_node_path(node)
            detail.update_detail(data, path)

    def _get_node_path(self, node: TreeNode) -> str:
        """Get the YAML path to a node."""
        parts = []
        current = node
        while current.parent is not None:
            data = current.data
            if isinstance(data, FieldSchema):
                parts.append(data.name)
            current = current.parent
        return ".".join(reversed(parts))

    def action_expand_all(self) -> None:
        """Expand all tree nodes."""
        tree = self.query_one(ConfigTree)
        tree.root.expand_all()

    def action_collapse_all(self) -> None:
        """Collapse all tree nodes."""
        tree = self.query_one(ConfigTree)
        for node in tree.root.children:
            node.collapse_all()

    def action_copy_yaml(self) -> None:
        """Copy YAML path of current selection."""
        tree = self.query_one(ConfigTree)
        if tree.cursor_node:
            path = self._get_node_path(tree.cursor_node)
            # Note: Clipboard access requires additional setup
            self.notify(f"Path: {path}")


def run_explorer(config_type: str = "benchmark") -> None:
    """Run the interactive config explorer."""
    app = ConfigExplorerApp(config_type)
    app.run()


if __name__ == "__main__":
    run_explorer()
