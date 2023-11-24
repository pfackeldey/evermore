from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from dilax.pdf import HashablePDF
from dilax.util import as1darray

__all__ = [
    "Parameter",
]


def __dir__():
    return __all__


class Parameter(eqx.Module):
    value: jax.Array = eqx.field(converter=as1darray)
    bounds: tuple[jax.Array, jax.Array] = eqx.field(
        static=True, converter=lambda x: tuple(map(as1darray, x))
    )
    constraints: set[HashablePDF] = eqx.field(static=True)

    def __init__(
        self,
        value: jax.Array,
        bounds: tuple[jax.Array, jax.Array] = (as1darray(-jnp.inf), as1darray(jnp.inf)),
    ) -> None:
        self.value = value
        self.bounds = bounds
        self.constraints: set[HashablePDF] = set()

    def update(self, value: jax.Array) -> Parameter:
        return eqx.tree_at(lambda t: t.value, self, value)

    @property
    def boundary_penalty(self) -> jax.Array:
        return jnp.where(
            (self.value < self.bounds[0]) | (self.value > self.bounds[1]),
            jnp.inf,
            0,
        )
