from __future__ import annotations

from typing import TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Scalar, Shaped

from evermore.binned.effect import Identity, OffsetAndScale
from evermore.binned.modifier import Modifier, ModifierBase, Where
from evermore.parameters.parameter import NormalParameter
from evermore.parameters.tree import PT
from evermore.util import maybe_float_array

__all__ = [
    "StatErrors",
]


def __dir__():
    return __all__


N = TypeVar("N", bound=Shaped[Array, "..."])


class StatErrors(ModifierBase[PT]):
    """
    Create staterror (barlow-beeston) parameters.

    Example:

    .. code-block:: python

        import jax.numpy as jnp
        import jax.tree_util as jtu
        import evermore as evm


        # histograms with bin contents and variances
        hists = {"qcd": jnp.array([10.0, 20.0, 30.0]), "signal": jnp.array([5.0, 10.0, 15.0])}
        histsw2 = {"qcd": jnp.array([12.0, 21.0, 29.0]), "signal": jnp.array([5.0, 8.0, 11.0])}

        # BB-full example:
        staterrors = jtu.tree_map(
            evm.staterror.StatErrors,
            hists,
            histsw2,
        )

        # apply it
        modified_qcd = staterrors["qcd"](hists["qcd"])
        modified_signal = staterrors["signal"](hists["signal"])

        # BB-lite example:
        staterrors = evm.staterror.StatErrors(
            evm.util.sum_over_leaves(hists),
            evm.util.sum_over_leaves(histsw2),
        )

        # apply it
        modified_qcd = staterrors(hists["qcd"])
        modified_signal = staterrors(hists["signal"])
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
        hist, variance = jax.tree.map(maybe_float_array, (hist, variance))

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

    def offset_and_scale(self, hist: Float[N]) -> OffsetAndScale[Float[N]]:
        modifier: Where[PT] = Where(
            self.non_empty_mask,
            self.parameter.scale(slope=self.relative_error, offset=1.0),
            Modifier(parameter=self.parameter, effect=Identity()),
        )
        return modifier.offset_and_scale(hist)
