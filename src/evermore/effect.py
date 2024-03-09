import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from evermore.custom_types import SF
from evermore.parameter import Parameter
from evermore.pdf import PDF, Flat, Gauss, Poisson
from evermore.util import as1darray

if TYPE_CHECKING:
    pass
else:
    pass


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
    @abc.abstractmethod
    def constraint(self, parameter: Parameter) -> PDF:
        ...

    @abc.abstractmethod
    def scale_factor(self, parameter: Parameter, sumw: Array) -> SF:
        ...


class unconstrained(Effect):
    def constraint(self, parameter: Parameter) -> PDF:
        return Flat()

    def scale_factor(self, parameter: Parameter, sumw: Array) -> SF:
        sf = jnp.broadcast_to(parameter.value, sumw.shape)
        return SF(multiplicative=sf, additive=jnp.zeros_like(sumw))


DEFAULT_EFFECT = unconstrained()


class gauss(Effect):
    width: Array = eqx.field(static=True, converter=as1darray)

    def __init__(self, width: Array) -> None:
        self.width = width

    def constraint(self, parameter: Parameter) -> PDF:
        return Gauss(
            mean=jnp.zeros_like(parameter.value), width=jnp.ones_like(parameter.value)
        )

    def scale_factor(self, parameter: Parameter, sumw: Array) -> SF:
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
        sf = jnp.broadcast_to((parameter.value * self.width) + 1, sumw.shape)
        return SF(multiplicative=sf, additive=jnp.zeros_like(sumw))


class shape(Effect):
    up: Array = eqx.field(converter=as1darray)
    down: Array = eqx.field(converter=as1darray)

    def __init__(
        self,
        up: Array,
        down: Array,
    ) -> None:
        self.up = up  # +1 sigma
        self.down = down  # -1 sigma

    def vshift(self, sf: Array, sumw: Array) -> Array:
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

    def constraint(self, parameter: Parameter) -> PDF:
        return Gauss(
            mean=jnp.zeros_like(parameter.value), width=jnp.ones_like(parameter.value)
        )

    def scale_factor(self, parameter: Parameter, sumw: Array) -> SF:
        sf = self.vshift(sf=parameter.value, sumw=sumw)
        return SF(multiplicative=jnp.ones_like(sumw), additive=sf)
        # shift = self.vshift(sf=sf, sumw=sumw)
        # # handle zeros, see: https://github.com/google/jax/issues/5039
        # x = jnp.where(sumw == 0.0, 1.0, sumw)
        # return jnp.where(sumw == 0.0, shift, (x + shift) / x)


class lnN(Effect):
    width: Float[Array, "2"] = eqx.field(static=True)

    def __init__(
        self,
        width: Float[Array, "2"],  # given as (down, up)
    ) -> None:
        assert width.shape == (2,)
        self.width = width

    def interpolate(self, parameter: Parameter) -> Array:
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

    def constraint(self, parameter: Parameter) -> PDF:
        return Gauss(
            mean=jnp.zeros_like(parameter.value), width=jnp.ones_like(parameter.value)
        )

    def scale_factor(self, parameter: Parameter, sumw: Array) -> SF:
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
        interp = self.interpolate(parameter=parameter)
        sf = jnp.broadcast_to(jnp.exp(parameter.value * interp), sumw.shape)
        return SF(multiplicative=sf, additive=jnp.zeros_like(sumw))


class poisson(Effect):
    lamb: Array = eqx.field(static=True, converter=as1darray)

    def __init__(self, lamb: Array) -> None:
        self.lamb = lamb

    def constraint(self, parameter: Parameter) -> PDF:
        assert parameter.value.shape == self.lamb.shape
        return Poisson(lamb=self.lamb)

    def scale_factor(self, parameter: Parameter, sumw: Array) -> SF:
        sf = jnp.broadcast_to(parameter.value + 1, sumw.shape)
        return SF(multiplicative=sf, additive=jnp.zeros_like(sumw))
