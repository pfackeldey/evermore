from __future__ import annotations

import dataclasses
import threading
from typing import Any

import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from evermore.custom_types import ModifierLike, PDFLike
from evermore.effect import (
    AsymmetricExponential,
    Effect,
    Identity,
    Linear,
    VerticalTemplateMorphing,
)
from evermore.modifier import (
    BooleanMask,
    Compose,
    Modifier,
    Transform,
    TransformOffset,
    TransformScale,
    Where,
)
from evermore.parameter import NormalParameter, Parameter
from evermore.pdf import Normal, Poisson

__all__ = [
    "convert_tree_to_penzai",
]


def __dir__():
    return __all__


@dataclasses.dataclass
class EvermoreClassesContext(threading.local):
    cls_types: list[Any] = dataclasses.field(default_factory=list)


Context = EvermoreClassesContext()


Context.cls_types.extend(
    [
        NormalParameter,
        Parameter,
        Identity,
        Linear,
        AsymmetricExponential,
        VerticalTemplateMorphing,
        Effect,
        Modifier,
        Compose,
        Where,
        BooleanMask,
        Transform,
        TransformScale,
        TransformOffset,
        Normal,
        Poisson,
        ModifierLike,
        PDFLike,
    ]
)


def convert_tree_to_penzai(tree: PyTree) -> PyTree:
    from functools import partial

    for cls in Context.cls_types:

        def _is_evm_cls(leaf: Any, cls: Any) -> bool:
            return isinstance(leaf, cls)

        tree = jtu.tree_map(
            partial(_convert, cls=cls), tree, is_leaf=partial(_is_evm_cls, cls=cls)
        )
    return tree


def _convert(leaf: Any, cls: Any) -> Any:
    from penzai import pz

    if isinstance(leaf, cls) and dataclasses.is_dataclass(leaf):
        fields = dataclasses.fields(leaf)

        leaf_cls = type(leaf)
        attributes: dict[str, Any] = {
            "__annotations__": {field.name: field.type for field in fields}
        }

        if callable(leaf_cls):
            attributes["__call__"] = leaf_cls.__call__

        def _pretty(x: Any) -> Any:
            if isinstance(x, Array) and x.size == 1:
                return x.item()
            return x

        attrs = {k: _pretty(getattr(leaf, k)) for k in attributes["__annotations__"]}

        new_cls = pz.pytree_dataclass(
            type(leaf_cls.__name__, (pz.Layer,), dict(attributes))
        )
        return new_cls(**attrs)
    return leaf
