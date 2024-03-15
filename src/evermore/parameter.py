from __future__ import annotations

from typing import TYPE_CHECKING, cast

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, PyTree

from evermore.custom_types import Sentinel, _NoValue
from evermore.pdf import PDF
from evermore.util import as1darray

if TYPE_CHECKING:
    from evermore.modifier import Modifier

__all__ = [
    "Parameter",
    "staterrors",
    "auto_init",
]


def __dir__():
    return __all__


class Parameter(eqx.Module):
    value: Array = eqx.field(converter=as1darray)
    lower: Array = eqx.field(converter=as1darray)
    upper: Array = eqx.field(converter=as1darray)
    constraint: PDF | Sentinel = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike = 0.0,
        lower: ArrayLike = -jnp.inf,
        upper: ArrayLike = jnp.inf,
        constraint: PDF | Sentinel = _NoValue,
    ) -> None:
        self.value = as1darray(value)
        self.lower = as1darray(lower)
        self.upper = as1darray(upper)
        self.constraint = constraint

    def _set_constraint(self, constraint: PDF, overwrite: bool = False) -> PDF:
        # Frozen dataclasses don't support setting attributes so we have to
        # overload that operation here as they do in the dataclass implementation
        assert isinstance(constraint, PDF)

        # If no constraint is set or overwriting is allowed, set it and return.
        if self.constraint is _NoValue or overwrite:
            object.__setattr__(self, "constraint", constraint)
            return constraint

        # Check if new constraint is compatible by class only, otherwise complain.
        # This is ok because we know that the constraints from evm.modifiers
        # will always be compatible within the same class (underlying arrays are equal by construction).
        # This significantly speeds up this check.
        if self.constraint.__class__ is not constraint.__class__:
            msg = f"Parameter constraint '{self.constraint}' is different from the new constraint '{constraint}'."
            raise ValueError(msg)
        return cast(PDF, self.constraint)

    @property
    def boundary_penalty(self) -> Array:
        return jnp.where(
            (self.value < self.lower) | (self.value > self.upper),
            jnp.inf,
            0,
        )

    # shorthands
    def unconstrained(self) -> Modifier:
        import evermore as evm

        return evm.Modifier(parameter=self, effect=evm.effect.unconstrained())

    def gauss(self, width: Array) -> Modifier:
        import evermore as evm

        return evm.Modifier(parameter=self, effect=evm.effect.gauss(width=width))

    def lnN(self, up: Array, down: Array) -> Modifier:
        import evermore as evm

        return evm.Modifier(parameter=self, effect=evm.effect.lnN(up=up, down=down))

    def poisson(self, lamb: Array) -> Modifier:
        import evermore as evm

        return evm.Modifier(parameter=self, effect=evm.effect.poisson(lamb=lamb))

    def shape(self, up: Array, down: Array) -> Modifier:
        import evermore as evm

        return evm.Modifier(parameter=self, effect=evm.effect.shape(up=up, down=down))


def staterrors(hists: PyTree[Array]) -> PyTree[Parameter]:
    """
    Create staterror (barlow-beeston) parameters.

    Example:

    .. code-block:: python

        import jax.numpy as jnp
        import evermore as evm

        hists = {"qcd": jnp.array([1, 2, 3]), "dy": jnp.array([4, 5, 6])}

        # bulk create staterrors
        staterrors = evm.parameter.staterrors(hists=hists)
    """

    leaves = jtu.tree_leaves(hists)
    # create parameters
    return {
        # per process and bin
        "poisson": jtu.tree_map(lambda _: Parameter(value=0.0), hists),
        # only per bin
        "gauss": Parameter(value=jnp.zeros_like(leaves[0])),
    }


def auto_init(module: eqx.Module) -> eqx.Module:
    import dataclasses
    import typing

    type_hints = typing.get_type_hints(module.__class__)
    for field in dataclasses.fields(module):
        name = field.name
        hint = type_hints[name]
        if issubclass(hint, Parameter) and not hasattr(module, name):
            setattr(module, name, hint())
    return module
