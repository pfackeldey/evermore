from __future__ import annotations

from typing import TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Scalar, Shaped

from evermore.binned.effect import Identity, OffsetAndScale
from evermore.binned.modifier import Modifier, ModifierBase, Where
from evermore.parameters.parameter import NormalParameter
from evermore.util import float_array

__all__ = [
    "StatErrors",
]


def __dir__():
    return __all__


N = TypeVar("N", bound=Shaped[Array, "..."])


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
    n_entries: Float[N]
    non_empty_mask: Bool[N]
    relative_error: Float[N]
    parameter: NormalParameter[Float[N]]

    def __init__(
        self,
        hist: Float[N],
        variance: Float[N],
    ):
        # make sure they are of dtype float
        hist, variance = jax.tree.map(float_array, (hist, variance))

        self.eps = jnp.finfo(variance.dtype).eps

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

    def offset_and_scale(self, hist: Float[N]) -> OffsetAndScale:
        modifier = Where(
            self.non_empty_mask,
            self.parameter.scale(slope=self.relative_error, offset=1.0),
            Modifier(parameter=self.parameter, effect=Identity()),
        )
        return modifier.offset_and_scale(hist)
