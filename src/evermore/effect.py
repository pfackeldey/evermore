import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from evermore.custom_types import SF
from evermore.parameter import Parameter
from evermore.util import as1darray

__all__ = [
    "Effect",
    "unconstrained",
    "normal",
    "log_normal",
    "poisson",
    "shape",
]


def __dir__():
    return __all__


class Effect(eqx.Module):
    @abc.abstractmethod
    def scale_factor(self, parameter: Parameter, hist: Array) -> SF: ...


class unconstrained(Effect):
    def scale_factor(self, parameter: Parameter, hist: Array) -> SF:
        sf = jnp.broadcast_to(parameter.value, hist.shape)
        return SF(multiplicative=sf, additive=jnp.zeros_like(hist))


DEFAULT_EFFECT = unconstrained()


class normal(Effect):
    width: Array = eqx.field(converter=as1darray)

    def scale_factor(self, parameter: Parameter, hist: Array) -> SF:
        """
        Implementation with (inverse) CDFs is defined as follows:

            .. code-block:: python

                gx = Normal(mean=1.0, width=self.width)  # type: ignore[arg-type]
                g1 = Normal(mean=1.0, width=1.0)

                return gx.inv_cdf(g1.cdf(parameter.value + 1))

        But we can use the fast analytical solution instead:

            .. code-block:: python

                return (parameter.value * self.width) + 1

        """
        sf = jnp.broadcast_to((parameter.value * self.width) + 1, hist.shape)
        return SF(multiplicative=sf, additive=jnp.zeros_like(hist))


class shape(Effect):
    up: Array = eqx.field(converter=as1darray)  # + 1 sigma
    down: Array = eqx.field(converter=as1darray)  # - 1 sigma

    def vshift(self, sf: Array, hist: Array) -> Array:
        factor = sf
        dx_sum = self.up + self.down - 2 * hist
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

    def scale_factor(self, parameter: Parameter, hist: Array) -> SF:
        sf = self.vshift(sf=parameter.value, hist=hist)
        return SF(multiplicative=jnp.ones_like(hist), additive=sf)


class log_normal(Effect):
    up: Array = eqx.field(converter=as1darray)
    down: Array = eqx.field(converter=as1darray)

    def interpolate(self, parameter: Parameter) -> Array:
        # https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/be488af288361ef101859a398ae618131373cad7/src/ProcessNormalization.cc#L112-L129
        x = parameter.value
        lo, hi = self.down, self.up
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

    def scale_factor(self, parameter: Parameter, hist: Array) -> SF:
        """
        Implementation with (inverse) CDFs is defined as follows:

            .. code-block:: python

                gx = Normal(mean=jnp.exp(parameter.value), width=width)  # type: ignore[arg-type]
                g1 = Normal(mean=1.0, width=1.0)

                return gx.inv_cdf(g1.cdf(parameter.value + 1))

        But we can use the fast analytical solution instead:

            .. code-block:: python

                return jnp.exp(parameter.value * self.interpolate(parameter=parameter))

        """
        interp = self.interpolate(parameter=parameter)
        sf = jnp.broadcast_to(jnp.exp(parameter.value * interp), hist.shape)
        return SF(multiplicative=sf, additive=jnp.zeros_like(hist))


class poisson(Effect):
    lamb: Array = eqx.field(converter=as1darray)

    def scale_factor(self, parameter: Parameter, hist: Array) -> SF:
        sf = jnp.broadcast_to(parameter.value + 1, hist.shape)
        return SF(multiplicative=sf, additive=jnp.zeros_like(hist))
