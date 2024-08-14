from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from treescope import (
    dataclass_util,
    formatting_util,
    renderers,
    rendering_parts,
)


class SupportsTreescope:
    def __treescope_repr__(
        self,
        path: str,
        subtree_renderer: Callable[[Any, str | None], rendering_parts.Rendering],
    ) -> rendering_parts.Rendering:
        return handle_evermore_classes(self, path, subtree_renderer)


def handle_evermore_classes(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> rendering_parts.RenderableTreePart | rendering_parts.Rendering:
    """Renders evermore classes.
    Taken from: https://github.com/google-deepmind/penzai/blob/b1bd577dc34f0e7b8f7fef3bbeb2cd571c2f8fcd/penzai/core/_treescope_handlers/struct_handler.py

    Args:
        node: The node to render.
        path: The path to the node. (Optional)
        subtree_renderer: A recursive renderer for subtrees.

    Returns:
        A rendering of evermore classes.
    """

    # get prefix, e.g. "Parameter("
    prefix = render_evermore_constructor(node)

    # get fields of the dataclass, e.g. value=1.0
    fields = dataclasses.fields(node)

    # get children of the tree
    children = rendering_parts.build_field_children(
        node,
        path,
        subtree_renderer,
        fields_or_attribute_names=fields,
        attr_style_fn=evermore_attr_style_fn_for_fields(fields),
    )

    # get colors for the background of the tree node
    def _treescope_color(node) -> str:
        """Returns the color of the tree node."""

        type_string = type(node).__module__ + "." + type(node).__qualname__
        return formatting_util.color_from_string(type_string)

    background_color, background_pattern = (
        formatting_util.parse_simple_color_and_pattern_spec(
            _treescope_color(node), type(node).__name__
        )
    )

    return rendering_parts.build_foldable_tree_node_from_children(
        prefix=prefix,
        children=children,
        suffix=")",
        background_color=background_color,
        background_pattern=background_pattern,
    )


def evermore_attr_style_fn_for_fields(
    fields,
) -> Callable[[str], rendering_parts.RenderableTreePart]:
    """Builds a function to render attributes of an evermore class.

    The resulting function will render pytree node fields in a different style.
    E.g. the field "value" of a Parameter class will be rendered in a different style.

    Taken from: https://github.com/google-deepmind/penzai/blob/b1bd577dc34f0e7b8f7fef3bbeb2cd571c2f8fcd/penzai/core/_treescope_handlers/struct_handler.py

    Args:
        fields: The fields of the evermore class.

    Returns:
        A function that takes a field name and returns a RenderableTreePart."""
    fields_by_name = {field.name: field for field in fields}

    def attr_style_fn(field_name):
        field = fields_by_name[field_name]
        if field.metadata.get("pytree_node", True):
            return rendering_parts.custom_style(
                rendering_parts.text(field_name),
                css_style="font-style: italic; color: #00255f;",
            )
        return rendering_parts.text(field_name)

    return attr_style_fn


def render_evermore_constructor(node: Any) -> rendering_parts.RenderableTreePart:
    """Renders the constructor of an evermore class, with an open parenthesis.
    Taken from: https://github.com/google-deepmind/penzai/blob/b1bd577dc34f0e7b8f7fef3bbeb2cd571c2f8fcd/penzai/core/_treescope_handlers/struct_handler.py
    """
    if dataclass_util.init_takes_fields(type(node)):
        return rendering_parts.siblings(
            rendering_parts.maybe_qualified_type_name(type(node)), "("
        )

    return rendering_parts.siblings(
        rendering_parts.maybe_qualified_type_name(type(node)),
        rendering_parts.roundtrip_condition(
            roundtrip=rendering_parts.text(".from_attributes")
        ),
    )
