from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from evermore.parameters.parameter import Parameter
from evermore.util import float_array
from evermore.visualization import SupportsTreescope

__all__ = [
    "AsymmetricExponential",
    "Effect",
    "Identity",
    "Linear",
    "VerticalTemplateMorphing",
]


def __dir__():
    return __all__


class OffsetAndScale(eqx.Module):
    offset: Float[Array, "..."] = eqx.field(converter=float_array, default=0.0)  # noqa: UP037
    scale: Float[Array, "..."] = eqx.field(converter=float_array, default=1.0)  # noqa: UP037

    def broadcast(self) -> OffsetAndScale:
        shape = jnp.broadcast_shapes(self.offset.shape, self.scale.shape)
        return type(self)(
            offset=jnp.broadcast_to(self.offset, shape),
            scale=jnp.broadcast_to(self.scale, shape),
        )


class Effect(eqx.Module, SupportsTreescope):
    @abc.abstractmethod
    def __call__(
        self,
        parameter: PyTree[Parameter],
        hist: Float[Array, "..."],  # noqa: UP037
    ) -> OffsetAndScale: ...


class Identity(Effect):
    @jax.named_scope("evm.effect.Identity")
    def __call__(
        self,
        parameter: PyTree[Parameter],
        hist: Float[Array, "..."],  # noqa: UP037
    ) -> OffsetAndScale:
        return OffsetAndScale(offset=0.0, scale=1.0)  # type: ignore[arg-type]


class Lambda(Effect):
    fun: Callable[
        [PyTree[Parameter], Float[Array, "..."]], OffsetAndScale | Float[Array, "..."]  # noqa: UP037
    ]
    normalize_by: Literal["offset", "scale"] | None = eqx.field(
        static=True, default=None
    )

    @jax.named_scope("evm.effect.Lambda")
    def __call__(
        self,
        parameter: PyTree[Parameter],
        hist: Float[Array, "..."],  # noqa: UP037
    ) -> OffsetAndScale:
        res = self.fun(parameter, hist)
        if isinstance(res, OffsetAndScale) and self.normalize_by is None:
            return res
        if self.normalize_by == "offset":
            return OffsetAndScale(offset=(res - hist), scale=1.0)  # type: ignore[arg-type]
        if self.normalize_by == "scale":
            return OffsetAndScale(offset=0.0, scale=(res / hist))  # type: ignore[arg-type]
        msg = f"Unknown normalization type '{self.normalize_by}' for '{res}'"
        raise ValueError(msg)


class Linear(Effect):
    offset: Float[Array, "..."] = eqx.field(converter=float_array)  # noqa: UP037
    slope: Float[Array, "..."] = eqx.field(converter=float_array)  # noqa: UP037

    @jax.named_scope("evm.effect.Linear")
    def __call__(
        self,
        parameter: PyTree[Parameter],
        hist: Float[Array, "..."],  # noqa: UP037
    ) -> OffsetAndScale:
        assert isinstance(parameter, Parameter)
        sf = parameter.value * self.slope + self.offset
        return OffsetAndScale(offset=0.0, scale=sf)  # type: ignore[arg-type]


DEFAULT_EFFECT: Linear = Linear(offset=0.0, slope=1.0)  # type: ignore[arg-type]


class VerticalTemplateMorphing(Effect):
    # + 1 sigma
    up_template: Float[Array, "..."] = eqx.field(converter=float_array)  # noqa: UP037
    # - 1 sigma
    down_template: Float[Array, "..."] = eqx.field(converter=float_array)  # noqa: UP037

    def vshift(
        self,
        x: Float[Array, "..."],  # noqa: UP037
        hist: Float[Array, "..."],  # noqa: UP037
    ) -> Float[Array, "..."]:  # noqa: UP037
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
    def __call__(
        self,
        parameter: PyTree[Parameter],
        hist: Float[Array, "..."],  # noqa: UP037
    ) -> OffsetAndScale:
        assert isinstance(parameter, Parameter)
        offset = self.vshift(parameter.value, hist=hist)
        return OffsetAndScale(offset=offset, scale=jnp.ones_like(hist))  # type: ignore[arg-type]


class AsymmetricExponential(Effect):
    up: Float[Array, "..."] = eqx.field(converter=float_array)  # noqa: UP037
    down: Float[Array, "..."] = eqx.field(converter=float_array)  # noqa: UP037

    def interpolate(self, x: Float[Array, "..."]) -> Float[Array, "..."]:  # noqa: UP037
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
    def __call__(
        self,
        parameter: PyTree[Parameter],
        hist: Float[Array, "..."],  # noqa: UP037
    ) -> OffsetAndScale:
        assert isinstance(parameter, Parameter)
        interp = self.interpolate(parameter.value)
        return OffsetAndScale(offset=0.0, scale=jnp.exp(parameter.value * interp))  # type: ignore[arg-type]
