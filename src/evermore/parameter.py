from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, PyTree

from evermore.custom_types import Sentinel, _NoValue
from evermore.pdf import PDF, Flat, Gauss, Poisson
from evermore.util import as1darray

if TYPE_CHECKING:
    from evermore.effect import Effect
    from evermore.modifier import Modifier

__all__ = [
    "Parameter",
    "FreeFloating",
    "GaussConstrained",
    "PoissonConstrained",
    "staterrors",
    "auto_init",
]


def __dir__():
    return __all__


class Parameter(eqx.Module):
    value: Array = eqx.field(converter=as1darray)
    lower: Array = eqx.field(static=True, converter=as1darray)
    upper: Array = eqx.field(static=True, converter=as1darray)
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

    def is_valid_effect(self, effect: Effect) -> bool:
        return True


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

    def is_valid_effect(self, effect: Effect) -> bool:
        import evermore as evm

        return isinstance(effect, evm.effect.unconstrained)


class GaussConstrained(Parameter):
    def __init__(
        self,
        value: ArrayLike = 0.0,
        lower: ArrayLike = -jnp.inf,
        upper: ArrayLike = jnp.inf,
    ) -> None:
        constraint = Gauss(
            mean=jnp.zeros_like(as1darray(value)), width=jnp.ones_like(as1darray(value))
        )
        super().__init__(value=value, lower=lower, upper=upper, constraint=constraint)

    def gauss(self, width: Array) -> Modifier:
        import evermore as evm

        return evm.Modifier(parameter=self, effect=evm.effect.gauss(width=width))

    def lnN(self, up: Array, down: Array) -> Modifier:
        import evermore as evm

        return evm.Modifier(parameter=self, effect=evm.effect.lnN(up=up, down=down))

    def shape(self, up: Array, down: Array) -> Modifier:
        import evermore as evm

        return evm.Modifier(parameter=self, effect=evm.effect.shape(up=up, down=down))

    def is_valid_effect(self, effect: Effect) -> bool:
        import evermore as evm

        return isinstance(effect, evm.effect.gauss | evm.effect.lnN | evm.effect.shape)


class PoissonConstrained(Parameter):
    hist: Array = eqx.field(converter=as1darray)

    def __init__(
        self,
        hist: ArrayLike,
        value: ArrayLike = 0.0,
        lower: ArrayLike = -jnp.inf,
        upper: ArrayLike = jnp.inf,
    ) -> None:
        self.hist = as1darray(hist)
        super().__init__(
            value=value, lower=lower, upper=upper, constraint=Poisson(lamb=self.hist)
        )

    def poisson(self) -> Modifier:
        import evermore as evm

        return evm.Modifier(parameter=self, effect=evm.effect.poisson(lamb=self.hist))

    def is_valid_effect(self, effect: Effect) -> bool:
        import evermore as evm

        return isinstance(effect, evm.effect.poisson)


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
        "poisson": jtu.tree_map(lambda hist: PoissonConstrained(hist=hist), hists),
        # only per bin
        "gauss": GaussConstrained(value=jnp.zeros_like(leaves[0])),
    }


def auto_init(module: eqx.Module) -> eqx.Module:
    import dataclasses
    import typing

    type_hints = typing.get_type_hints(module.__class__)
    for field in dataclasses.fields(module):
        name = field.name
        hint = type_hints[name]
        # we only have reasonable defaults for `FreeFloating` and `GaussConstrained`
        if issubclass(hint, FreeFloating | GaussConstrained) and not hasattr(
            module, name
        ):
            setattr(module, name, hint())
    return module
