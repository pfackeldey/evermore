from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Scalar

from evermore.binned.effect import H, Identity, OffsetAndScale
from evermore.binned.modifier import Modifier, ModifierBase, Where
from evermore.parameters.parameter import NormalParameter
from evermore.util import float_array

__all__ = [
    "StatErrors",
]


def __dir__():
    return __all__


class StatErrors(ModifierBase):
    """Creates per-bin Barlow-Beeston statistical uncertainty modifiers.

    Args:
        hist: Nominal histogram bin contents.
        variance: Estimated variance per bin.

    Examples:
        >>> import jax.numpy as jnp
        >>> import evermore as evm
        >>> hist = jnp.array([10.0, 20.0, 30.0])
        >>> var = jnp.array([12.0, 21.0, 29.0])
        >>> staterrors = evm.staterror.StatErrors(hist, var)
        >>> staterrors(hist)
        Array([10.        , 20.        , 29.999998], dtype=float32)
    """

    eps: Float[Scalar, ""]
    n_entries: Float[Array, "..."]  # noqa: UP037
    non_empty_mask: Bool[Array, "..."]  # noqa: UP037
    relative_error: Float[Array, "..."]  # noqa: UP037
    parameter: NormalParameter[Float[Array, "..."]]  # noqa: UP037

    def __init__(
        self,
        hist: Float[Array, "..."],  # noqa: UP037
        variance: Float[Array, "..."],  # noqa: UP037
    ):
        # make sure they are of dtype float
        hist, variance = jax.tree.map(float_array, (hist, variance))

        self.eps = cast(Float[Scalar, ""], jnp.finfo(variance.dtype).eps)

        self.n_entries = jnp.where(
            variance != 0.0,
            (hist**2 / (variance + jnp.where(variance != 0.0, 0.0, self.eps))),
            0.0,
        )
        self.non_empty_mask = self.n_entries != 0.0
        self.relative_error = jnp.where(
            self.non_empty_mask,
            1.0
            / jnp.sqrt(self.n_entries + jnp.where(self.non_empty_mask, 0.0, self.eps)),
            1.0,
        )
        self.parameter = NormalParameter(jnp.zeros_like(self.n_entries))

    def offset_and_scale(self, hist: H) -> OffsetAndScale:
        modifier = Where(
            self.non_empty_mask,
            self.parameter.scale(slope=self.relative_error, offset=1.0),
            Modifier(value=self.parameter.get_value(), effect=Identity()),
        )
        return modifier.offset_and_scale(hist)
