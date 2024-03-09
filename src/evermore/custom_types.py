from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from jaxtyping import Array

if TYPE_CHECKING:
    from evermore.modifier import compose


AddOrMul = Callable[[Array, Array], Array]
AddOrMulSFs = dict[AddOrMul, Array]


class Sentinel:
    repr: str

    def __init__(self, repr: str) -> None:
        self.repr = repr

    def __repr__(self) -> str:
        return self.repr

    __str__ = __repr__


_NoValue: Any = Sentinel("<NoValue>")


@runtime_checkable
class ModifierLike(Protocol):
    def scale_factor(self, sumw: Array) -> AddOrMulSFs:
        """
        Always return a dictionary of scale factors for the sumw array.
        Dictionary has to look as follows:

            .. code-block:: python

                import operator
                from jaxtyping import Array


                {operator.mul: Array, operator.add: Array}
        """
        ...

    def __call__(self, sumw: Array) -> Array:
        ...

    def __matmul__(self, other: ModifierLike) -> compose:
        ...
