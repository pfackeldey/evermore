import abc
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from evermore.custom_types import OffsetAndScale
from evermore.parameter import Parameter
from evermore.visualization import SupportsTreescope

__all__ = [
    "Effect",
    "Identity",
    "Linear",
    "VerticalTemplateMorphing",
    "AsymmetricExponential",
]


def __dir__():
    return __all__


class Effect(eqx.Module, SupportsTreescope):
    @abc.abstractmethod
    def __call__(self, parameter: PyTree[Parameter], hist: Array) -> OffsetAndScale: ...


class Identity(Effect):
    @jax.named_scope("evm.effect.Identity")
    def __call__(self, parameter: PyTree[Parameter], hist: Array) -> OffsetAndScale:
        return OffsetAndScale(offset=0.0, scale=1.0)


class Lambda(Effect):
    fun: Callable[[PyTree[Parameter], Array], OffsetAndScale | Array]
    normalize_by: Literal["offset", "scale"] | None = eqx.field(
        static=True, default=None
    )

    @jax.named_scope("evm.effect.Lambda")
    def __call__(self, parameter: PyTree[Parameter], hist: Array) -> OffsetAndScale:
        res = self.fun(parameter, hist)
        if isinstance(res, OffsetAndScale) and self.normalize_by is None:
            return res
        if isinstance(res, Array):
            if self.normalize_by == "offset":
                return OffsetAndScale(offset=(res - hist), scale=1.0)
            if self.normalize_by == "scale":
                return OffsetAndScale(offset=0.0, scale=(res / hist))
        msg = f"Unknown normalization type '{self.normalize_by}' for '{res}'"
        raise ValueError(msg)


class Linear(Effect):
    offset: Array = eqx.field(converter=jnp.atleast_1d)
    slope: Array = eqx.field(converter=jnp.atleast_1d)

    @jax.named_scope("evm.effect.Linear")
    def __call__(self, parameter: PyTree[Parameter], hist: Array) -> OffsetAndScale:
        assert isinstance(parameter, Parameter)
        sf = parameter.value * self.slope + self.offset
        return OffsetAndScale(offset=0.0, scale=sf)


DEFAULT_EFFECT: Linear = Linear(offset=0.0, slope=1.0)


class VerticalTemplateMorphing(Effect):
    up_template: Array = eqx.field(converter=jnp.atleast_1d)  # + 1 sigma
    down_template: Array = eqx.field(converter=jnp.atleast_1d)  # - 1 sigma

    def vshift(self, x: Array, hist: Array) -> Array:
        dx_sum = self.up_template + self.down_template - 2 * hist
        dx_diff = self.up_template - self.down_template

        # taken from https://github.com/nsmith-/jaxfit/blob/8479cd73e733ba35462287753fab44c0c560037b/src/jaxfit/roofit/combine.py#L173C6-L192
        _asym_poly = jnp.array([3.0, -10.0, 15.0, 0.0]) / 8.0

        abs_value = jnp.abs(x)
        return 0.5 * (
            dx_diff * x
            + dx_sum
            * jnp.where(
                abs_value > 1.0,
                abs_value,
                jnp.polyval(_asym_poly, x * x),
            )
        )

    @jax.named_scope("evm.effect.VerticalTemplateMorphing")
    def __call__(self, parameter: PyTree[Parameter], hist: Array) -> OffsetAndScale:
        assert isinstance(parameter, Parameter)
        offset = self.vshift(parameter.value, hist=hist)
        return OffsetAndScale(offset=offset, scale=1.0)


class AsymmetricExponential(Effect):
    up: Array = eqx.field(converter=jnp.atleast_1d)
    down: Array = eqx.field(converter=jnp.atleast_1d)

    def interpolate(self, x: Array) -> Array:
        # https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/be488af288361ef101859a398ae618131373cad7/src/ProcessNormalization.cc#L112-L129
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

    @jax.named_scope("evm.effect.AsymmetricExponential")
    def __call__(self, parameter: PyTree[Parameter], hist: Array) -> OffsetAndScale:
        assert isinstance(parameter, Parameter)
        interp = self.interpolate(parameter.value)
        return OffsetAndScale(offset=0.0, scale=jnp.exp(parameter.value * interp))
