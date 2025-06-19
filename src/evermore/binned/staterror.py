from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from evermore.binned.effect import Identity, OffsetAndScale
from evermore.binned.modifier import Modifier, ModifierBase, ModifierLike, Where
from evermore.parameters.parameter import NormalParameter
from evermore.util import atleast_1d_float_array

__all__ = [
    "StatErrors",
]


def __dir__():
    return __all__


class StatErrors(ModifierBase):
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

        # BB-lite example:
        staterrors = jtu.tree_map(
            evm.staterror.StatErrors,
            hists,
            histsw2,
        )

        # apply it
        modified_qcd = staterrors["qcd"](hists["qcd"])
        modified_signal = staterrors["signal"](hists["signal"])

        # BB-full example:
        staterrors = evm.staterror.StatErrors(
            evm.util.sum_over_leaves(hists),
            evm.util.sum_over_leaves(histsw2),
        )

        # apply it
        modified_qcd = staterrors(hists["qcd"])
        modified_signal = staterrors(hists["signal"])
    """

    modifier: ModifierLike

    def __init__(
        self,
        hist: Array,
        variance: Array,
    ):
        # make sure they are of dtype float
        hist, variance = jax.tree.map(atleast_1d_float_array, (hist, variance))

        eps = jnp.finfo(variance.dtype).eps

        n_entries = jnp.where(
            variance != 0.0,
            (hist**2 / (variance + jnp.where(variance != 0.0, 0.0, eps))),
            0.0,
        )
        parameter = NormalParameter(value=jnp.zeros_like(n_entries))

        non_empty_mask = n_entries != 0.0
        rel_err = jnp.where(
            non_empty_mask,
            1.0 / jnp.sqrt(n_entries + jnp.where(non_empty_mask, 0.0, eps)),
            1.0,
        )
        self.modifier = Where(
            non_empty_mask,
            parameter.scale(slope=rel_err, offset=1.0),
            Modifier(parameter=parameter, effect=Identity()),
        )

    def offset_and_scale(self, hist: Array) -> OffsetAndScale:
        return self.modifier.offset_and_scale(hist)
