from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from evermore.pdf import HashablePDF
from evermore.util import as1darray

if TYPE_CHECKING:
    from evermore.modifier import modifier

__all__ = [
    "Parameter",
    "auto_init",
]


def __dir__():
    return __all__


class Parameter(eqx.Module):
    value: Array = eqx.field(converter=as1darray)
    lower: Array = eqx.field(static=True, converter=as1darray)
    upper: Array = eqx.field(static=True, converter=as1darray)
    constraints: set[HashablePDF] = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike = 0.0,
        lower: ArrayLike = -jnp.inf,
        upper: ArrayLike = jnp.inf,
    ) -> None:
        self.value = as1darray(value)
        self.lower = as1darray(lower)
        self.upper = as1darray(upper)
        self.constraints: set[HashablePDF] = set()

    def update(self, value: Array | Parameter) -> Parameter:
        if isinstance(value, Parameter):
            value = value.value
        return eqx.tree_at(lambda t: t.value, self, value)

    @property
    def boundary_penalty(self) -> Array:
        return jnp.where(
            (self.value < self.lower) | (self.value > self.upper),
            jnp.inf,
            0,
        )

    # shorthands
    def unconstrained(self) -> modifier:
        import evermore as evm

        return evm.modifier(parameter=self, effect=evm.effect.unconstrained())

    def gauss(self, width: Array) -> modifier:
        import evermore as evm

        return evm.modifier(parameter=self, effect=evm.effect.gauss(width=width))

    def lnN(self, width: Float[Array, 2]) -> modifier:
        import evermore as evm

        return evm.modifier(parameter=self, effect=evm.effect.lnN(width=width))

    def poisson(self, lamb: Array) -> modifier:
        import evermore as evm

        return evm.modifier(parameter=self, effect=evm.effect.poisson(lamb=lamb))

    def shape(self, up: Array, down: Array) -> modifier:
        import evermore as evm

        return evm.modifier(parameter=self, effect=evm.effect.shape(up=up, down=down))


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
