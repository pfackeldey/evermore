from __future__ import annotations

import abc
import typing as tp
from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, ArrayLike, Float

from evermore.parameters.parameter import V
from evermore.util import float_array

__all__ = [
    "AsymmetricExponential",
    "BaseEffect",
    "Identity",
    "Lambda",
    "Linear",
    "OffsetAndScale",
    "SymmetricExponential",
    "VerticalTemplateMorphing",
]


def __dir__():
    return __all__


H = tp.TypeVar("H", bound=Float[Array, "..."])


class OffsetAndScale(nnx.Pytree):
    def __init__(self, offset: ArrayLike = 0.0, scale: ArrayLike = 1.0):
        self.offset = float_array(offset)
        self.scale = float_array(scale)

    def broadcast(self) -> OffsetAndScale:
        shape = jnp.broadcast_shapes(self.offset.shape, self.scale.shape)
        return type(self)(
            offset=jnp.broadcast_to(self.offset, shape),
            scale=jnp.broadcast_to(self.scale, shape),
        )


class BaseEffect(nnx.Module):
    @abc.abstractmethod
    def __call__(self, value: V, hist: H) -> OffsetAndScale: ...


class Identity(BaseEffect):
    def __call__(self, value: V, hist: H) -> OffsetAndScale:
        del value  # unused
        return OffsetAndScale(
            offset=jnp.zeros_like(hist), scale=jnp.ones_like(hist)
        ).broadcast()


class Lambda(BaseEffect):
    def __init__(
        self,
        fun: Callable[[V, H], OffsetAndScale | H],
        normalize_by: tp.Literal["offset", "scale"] | None = None,
    ):
        self.fun = fun
        self.normalize_by = normalize_by

    def __call__(self, value: V, hist: H) -> OffsetAndScale:
        res = self.fun(value, hist)  # ty:ignore[invalid-argument-type]
        if isinstance(res, OffsetAndScale):
            if self.normalize_by is not None:
                msg = f"Normalization not supported for OffsetAndScale return value from {self.fun}"
                raise ValueError(msg)
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


class Linear(BaseEffect):
    def __init__(self, offset: ArrayLike, slope: ArrayLike):
        self.offset = float_array(offset)
        self.slope = float_array(slope)

    def __call__(self, value: V, hist: H) -> OffsetAndScale:
        sf = value * self.slope + self.offset
        return OffsetAndScale(offset=jnp.zeros_like(hist), scale=sf).broadcast()


class VerticalTemplateMorphing(BaseEffect):
    def __init__(self, up_template: H, down_template: H):
        # + 1 sigma
        self.up_template: H = float_array(up_template)
        # - 1 sigma
        self.down_template: H = float_array(down_template)

    def vshift(self, value: V, hist: H) -> H:
        dx_sum = self.up_template + self.down_template - 2 * hist
        dx_diff = self.up_template - self.down_template

        # taken from https://github.com/nsmith-/jaxfit/blob/8479cd73e733ba35462287753fab44c0c560037b/src/jaxfit/roofit/combine.py#L173C6-L192
        _asym_poly = jnp.array([3.0, -10.0, 15.0, 0.0]) / 8.0

        abs_value = jnp.abs(value)
        return jnp.array(0.5) * (
            dx_diff * value
            + dx_sum
            * jnp.where(
                abs_value > 1.0,
                abs_value,
                jnp.polyval(_asym_poly, value * value),
            )
        )  # ty:ignore[invalid-return-type]

    def __call__(self, value: V, hist: H) -> OffsetAndScale:
        offset = self.vshift(value, hist=hist)
        return OffsetAndScale(offset=offset, scale=jnp.ones_like(hist)).broadcast()


class AsymmetricExponential(BaseEffect):
    def __init__(self, up: H, down: H):
        self.up: H = float_array(up)
        self.down: H = float_array(down)

    def interpolate(self, value: V) -> V:
        # https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/be488af288361ef101859a398ae618131373cad7/src/ProcessNormalization.cc#L112-L129
        lo, hi = self.down, self.up
        hi = jnp.log(hi)
        lo = jnp.log(lo)
        lo = -lo
        avg = 0.5 * (hi + lo)
        halfdiff = 0.5 * (hi - lo)
        twox = value + value
        twox2 = twox * twox
        alpha = 0.125 * twox * (twox2 * (3 * twox2 - 10.0) + 15.0)
        return jnp.where(
            jnp.abs(value) >= 0.5, jnp.where(value >= 0, hi, lo), avg + alpha * halfdiff
        )  # ty:ignore[invalid-return-type]

    def __call__(self, value: V, hist: H) -> OffsetAndScale:
        interp = self.interpolate(value)
        return OffsetAndScale(
            offset=jnp.zeros_like(hist), scale=jnp.exp(value * interp)
        ).broadcast()


class SymmetricExponential(BaseEffect):
    def __init__(self, kappa: H):
        self.kappa = float_array(kappa)

    def __call__(self, value: V, hist: H) -> OffsetAndScale:
        # https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/be488af288361ef101859a398ae618131373cad7/src/ProcessNormalization.cc#L90
        # scale with exp(log(kappa) * theta) = kappa^theta
        return OffsetAndScale(
            offset=jnp.zeros_like(hist),
            scale=self.kappa**value,
        ).broadcast()
