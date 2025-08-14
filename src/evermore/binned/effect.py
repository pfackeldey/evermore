from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Generic, Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from evermore.parameters.parameter import AbstractParameter
from evermore.parameters.tree import PT
from evermore.util import maybe_float_array
from evermore.visualization import SupportsTreescope

__all__ = [
    "AbstractEffect",
    "AsymmetricExponential",
    "Identity",
    "Linear",
    "VerticalTemplateMorphing",
]


def __dir__():
    return __all__


H = TypeVar("H", bound=Float[Array, "..."])


class OffsetAndScale(eqx.Module, Generic[H]):
    offset: H = eqx.field(converter=maybe_float_array, default=0.0)
    scale: H = eqx.field(converter=maybe_float_array, default=1.0)

    def broadcast(self) -> OffsetAndScale[H]:
        shape = jnp.broadcast_shapes(self.offset.shape, self.scale.shape)
        return type(self)(
            offset=jnp.broadcast_to(self.offset, shape),
            scale=jnp.broadcast_to(self.scale, shape),
        )


class AbstractEffect(eqx.Module, Generic[H], SupportsTreescope):
    @abc.abstractmethod
    def __call__(self, parameter: PT, hist: H) -> OffsetAndScale[H]: ...


class Identity(AbstractEffect[H]):
    @jax.named_scope("evm.effect.Identity")
    def __call__(self, parameter: PT, hist: H) -> OffsetAndScale[H]:
        return OffsetAndScale(
            offset=jnp.zeros_like(hist), scale=jnp.ones_like(hist)
        ).broadcast()


class Lambda(AbstractEffect[H]):
    fun: Callable[[PT, H], OffsetAndScale[H] | H]
    normalize_by: Literal["offset", "scale"] | None = eqx.field(
        static=True, default=None
    )

    @jax.named_scope("evm.effect.Lambda")
    def __call__(self, parameter: PT, hist: H) -> OffsetAndScale[H]:
        assert isinstance(parameter, AbstractParameter)
        res = self.fun(parameter, hist)
        if isinstance(res, OffsetAndScale) and self.normalize_by is None:
            return res
        if self.normalize_by == "offset":
            return OffsetAndScale(
                offset=(res - hist), scale=jnp.ones_like(hist)
            ).broadcast()
        if self.normalize_by == "scale":
            return OffsetAndScale(
                offset=jnp.zeros_like(hist), scale=(res / hist)
            ).broadcast()
        msg = f"Unknown normalization type '{self.normalize_by}' for '{res}'"
        raise ValueError(msg)


class Linear(AbstractEffect[H]):
    offset: H = eqx.field(converter=maybe_float_array)
    slope: H = eqx.field(converter=maybe_float_array)

    @jax.named_scope("evm.effect.Linear")
    def __call__(self, parameter: PT, hist: H) -> OffsetAndScale[H]:
        assert isinstance(parameter, AbstractParameter)
        sf = parameter.value * self.slope + self.offset
        return OffsetAndScale(offset=jnp.zeros_like(hist), scale=sf).broadcast()


class VerticalTemplateMorphing(AbstractEffect[H]):
    # + 1 sigma
    up_template: H = eqx.field(converter=maybe_float_array)
    # - 1 sigma
    down_template: H = eqx.field(converter=maybe_float_array)

    def vshift(self, x: H, hist: H) -> H:
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
    def __call__(self, parameter: PT, hist: H) -> OffsetAndScale[H]:
        assert isinstance(parameter, AbstractParameter)
        offset = self.vshift(parameter.value, hist=hist)
        return OffsetAndScale(offset=offset, scale=jnp.ones_like(hist)).broadcast()


class AsymmetricExponential(AbstractEffect[H]):
    up: H = eqx.field(converter=maybe_float_array)
    down: H = eqx.field(converter=maybe_float_array)

    def interpolate(self, x: H) -> H:
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
    def __call__(self, parameter: PT, hist: H) -> OffsetAndScale[H]:
        assert isinstance(parameter, AbstractParameter)
        interp = self.interpolate(parameter.value)
        return OffsetAndScale(
            offset=jnp.zeros_like(hist), scale=jnp.exp(parameter.value * interp)
        ).broadcast()
