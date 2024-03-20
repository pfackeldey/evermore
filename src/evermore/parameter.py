from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from evermore.custom_types import Sentinel, _NoValue
from evermore.pdf import PDF, Flat, Normal, Poisson
from evermore.util import as1darray

if TYPE_CHECKING:
    from evermore.modifier import Modifier

__all__ = [
    "Parameter",
    "FreeFloating",
    "NormalConstrained",
    "PoissonConstrained",
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
        value: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        constraint: PDF | Sentinel = _NoValue,
    ) -> None:
        self.value = as1darray(value)
        self.lower = jnp.broadcast_to(as1darray(lower), self.value.shape)
        self.upper = jnp.broadcast_to(as1darray(upper), self.value.shape)
        self.constraint = constraint

    @property
    def boundary_penalty(self) -> Array:
        return jnp.where(
            (self.value < self.lower) | (self.value > self.upper),
            jnp.inf,
            0,
        )


class FreeFloating(Parameter):
    def __init__(
        self,
        value: ArrayLike = 1.0,
        lower: ArrayLike = -jnp.inf,
        upper: ArrayLike = jnp.inf,
    ) -> None:
        super().__init__(value=value, lower=lower, upper=upper, constraint=Flat())

    def unconstrained(self) -> Modifier:
        import evermore as evm

        return evm.Modifier(parameter=self, effect=evm.effect.unconstrained())


class NormalConstrained(Parameter):
    def __init__(
        self,
        value: ArrayLike = 0.0,
        lower: ArrayLike = -jnp.inf,
        upper: ArrayLike = jnp.inf,
    ) -> None:
        super().__init__(
            value=value,
            lower=lower,
            upper=upper,
            constraint=Normal(
                mean=jnp.zeros_like(as1darray(value)),
                width=jnp.ones_like(as1darray(value)),
            ),
        )

    def normal(self, width: Array) -> Modifier:
        import evermore as evm

        return evm.Modifier(parameter=self, effect=evm.effect.normal(width=width))

    def log_normal(self, up: Array, down: Array) -> Modifier:
        import evermore as evm

        return evm.Modifier(
            parameter=self, effect=evm.effect.log_normal(up=up, down=down)
        )

    def shape(self, up: Array, down: Array) -> Modifier:
        import evermore as evm

        return evm.Modifier(parameter=self, effect=evm.effect.shape(up=up, down=down))


class PoissonConstrained(Parameter):
    lamb: Array = eqx.field(converter=as1darray)

    def __init__(
        self,
        lamb: ArrayLike,
        value: ArrayLike = 0.0,
        lower: ArrayLike = -jnp.inf,
        upper: ArrayLike = jnp.inf,
    ) -> None:
        self.lamb = as1darray(lamb)
        super().__init__(
            value=value,
            lower=lower,
            upper=upper,
            constraint=Poisson(lamb=self.lamb),
        )

    def poisson(self) -> Modifier:
        import evermore as evm

        return evm.Modifier(parameter=self, effect=evm.effect.poisson(lamb=self.lamb))


def auto_init(module: eqx.Module) -> eqx.Module:
    import dataclasses
    import typing

    type_hints = typing.get_type_hints(module.__class__)
    for field in dataclasses.fields(module):
        name = field.name
        hint = type_hints[name]
        # we only have reasonable defaults for `FreeFloating` and `NormalConstrained`
        if issubclass(hint, FreeFloating | NormalConstrained) and not hasattr(
            module, name
        ):
            setattr(module, name, hint())
    return module
