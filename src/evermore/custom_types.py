from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from jaxtyping import Array

if TYPE_CHECKING:
    from evermore.modifier import compose


__all__ = [
    "SF",
    "AddOrMul",
    "ModifierLike",
]


AddOrMul = Callable[[Array, Array], Array]


class SF(NamedTuple):
    multiplicative: Array
    additive: Array


class Sentinel:
    __slots__ = ("repr",)

    def __init__(self, repr: str) -> None:
        self.repr = repr

    def __repr__(self) -> str:
        return self.repr

    __str__ = __repr__


_NoValue: Any = Sentinel("<NoValue>")


class ModifierLike(Protocol):
    def scale_factor(self, hist: Array) -> SF:
        ...

    def __call__(self, hist: Array) -> Array:
        ...

    def __matmul__(self, other: ModifierLike) -> compose:
        ...
