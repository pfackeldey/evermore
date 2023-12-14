from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from dilax.pdf import Flat, Gauss, HashablePDF, Poisson
from dilax.util import as1darray

ArrayLike = jax.typing.ArrayLike

if TYPE_CHECKING:
    from dilax.parameter import Parameter

__all__ = [
    "Effect",
    "unconstrained",
    "gauss",
    "lnN",
    "poisson",
    "shape",
]


def __dir__():
    return __all__


class Effect(eqx.Module):
    @property
    @abc.abstractmethod
    def constraint(self) -> HashablePDF:
        ...

    @abc.abstractmethod
    def scale_factor(self, parameter: Parameter, sumw: jax.Array) -> jax.Array:
        ...


class unconstrained(Effect):
    @property
    def constraint(self) -> HashablePDF:
        return Flat()

    def scale_factor(self, parameter: Parameter, sumw: jax.Array) -> jax.Array:
        return parameter.value


DEFAULT_EFFECT = unconstrained()


class gauss(Effect):
    width: ArrayLike = eqx.field(static=True, converter=as1darray)

    def __init__(self, width: ArrayLike) -> None:
        self.width = width

    @property
    def constraint(self) -> HashablePDF:
        return Gauss(mean=0.0, width=1.0)

    def scale_factor(self, parameter: Parameter, sumw: jax.Array) -> jax.Array:
        """
        Implementation with (inverse) CDFs is defined as follows:

            .. code-block:: python

                gx = Gauss(mean=1.0, width=self.width)  # type: ignore[arg-type]
                g1 = Gauss(mean=1.0, width=1.0)

                return gx.inv_cdf(g1.cdf(parameter.value + 1))

        But we can use the fast analytical solution instead:

            .. code-block:: python

                return (parameter.value * self.width) + 1

        """
        return (parameter.value * self.width) + 1


class shape(Effect):
    up: jax.Array = eqx.field(converter=as1darray)
    down: jax.Array = eqx.field(converter=as1darray)

    def __init__(
        self,
        up: jax.Array,
        down: jax.Array,
    ) -> None:
        self.up = up  # +1 sigma
        self.down = down  # -1 sigma

    def vshift(self, sf: jax.Array, sumw: jax.Array) -> jax.Array:
        factor = sf
        dx_sum = self.up + self.down - 2 * sumw
        dx_diff = self.up - self.down

        # taken from https://github.com/nsmith-/jaxfit/blob/8479cd73e733ba35462287753fab44c0c560037b/src/jaxfit/roofit/combine.py#L173C6-L192
        _asym_poly = jnp.array([3.0, -10.0, 15.0, 0.0]) / 8.0

        abs_value = jnp.abs(factor)
        return 0.5 * (
            dx_diff * factor
            + dx_sum
            * jnp.where(
                abs_value > 1.0,
                abs_value,
                jnp.polyval(_asym_poly, factor * factor),
            )
        )

    @property
    def constraint(self) -> HashablePDF:
        return Gauss(mean=0.0, width=1.0)

    def scale_factor(self, parameter: Parameter, sumw: jax.Array) -> jax.Array:
        sf = parameter.value
        shift = self.vshift(sf=sf, sumw=sumw)
        # handle zeros, see: https://github.com/google/jax/issues/5039
        x = jnp.where(sumw == 0.0, 1.0, sumw)
        return jnp.where(sumw == 0.0, 1.0, (x + shift) / x)


class lnN(Effect):
    width: tuple[ArrayLike, ArrayLike] = eqx.field(static=True)

    def __init__(
        self,
        width: tuple[ArrayLike, ArrayLike],
    ) -> None:
        self.width = width

    def interpolate(self, parameter: Parameter) -> jax.Array:
        # https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/be488af288361ef101859a398ae618131373cad7/src/ProcessNormalization.cc#L112-L129
        x = parameter.value
        lo, hi = self.width
        hi = jnp.log(hi)
        lo = jnp.log(lo)
        lo = -lo
        avg = 0.5 * (hi + lo)
        halfdiff = 0.5 * (hi - lo)
        twox = x + x
        twox2 = twox * twox
        alpha = 0.125 * twox * (twox2 * (3 * twox2 - 10.0) + 15.0)
        return jnp.where(
            jnp.abs(x) >= 0.5, jnp.where(x >= 0, hi, lo), avg + alpha * halfdiff
        )

    @property
    def constraint(self) -> HashablePDF:
        return Gauss(mean=0.0, width=1.0)

    def scale_factor(self, parameter: Parameter, sumw: jax.Array) -> jax.Array:
        """
        Implementation with (inverse) CDFs is defined as follows:

            .. code-block:: python

                gx = Gauss(mean=jnp.exp(parameter.value), width=width)  # type: ignore[arg-type]
                g1 = Gauss(mean=1.0, width=1.0)

                return gx.inv_cdf(g1.cdf(parameter.value + 1))

        But we can use the fast analytical solution instead:

            .. code-block:: python

                return jnp.exp(parameter.value * self.interpolate(parameter=parameter))

        """
        return jnp.exp(parameter.value * self.interpolate(parameter=parameter))


class poisson(Effect):
    lamb: jax.Array = eqx.field(static=True, converter=as1darray)

    def __init__(self, lamb: jax.Array) -> None:
        self.lamb = lamb

    @property
    def constraint(self) -> HashablePDF:
        return Poisson(lamb=self.lamb)

    def scale_factor(self, parameter: Parameter, sumw: jax.Array) -> jax.Array:
        return parameter.value + 1
