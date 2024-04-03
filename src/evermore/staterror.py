from __future__ import annotations

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from evermore.custom_types import ModifierLike
from evermore.effect import noop
from evermore.modifier import Modifier
from evermore.modifier import where as modifier_where
from evermore.parameter import NormalConstrained, PoissonConstrained
from evermore.util import sum_leaves

__all__ = [
    "StatErrors",
]


def __dir__():
    return __all__


class StatErrors(eqx.Module):
    """
    Create staterror (barlow-beeston) parameters.

    Example:

        .. code-block:: python

            import jax.numpy as jnp
            import evermore as evm

            hists = {"qcd": jnp.array([10, 20, 30]), "signal": jnp.array([5, 10, 15])}
            histsw2 = {"qcd": jnp.array([12, 21, 29]), "signal": jnp.array([5, 8, 11])}

            staterrors = evm.staterror.StatErrors(hists, histsw2, threshold=10.0)

            # Create a modifier for the qcd process, `where` is a function
            # that finds the corresponding parameter from `staterrors.params_per_process`
            mod = staterrors.get(where=lambda x: x["qcd"])
            # apply the modifier to the parameter
            mod(hists["qcd"])
    """

    gaussians_global: PyTree
    gaussians_per_process: PyTree
    poissons_per_process: PyTree
    hists: PyTree = eqx.field(static=True)
    histsw2: PyTree = eqx.field(static=True)
    ntot: Array = eqx.field(static=True)
    etot: Array = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    mask: Array = eqx.field(static=True)

    def __init__(
        self,
        hists: PyTree,
        histsw2: PyTree,
        threshold: float = 10.0,
    ) -> None:
        assert (
            jtu.tree_structure(hists) == jtu.tree_structure(histsw2)  # type: ignore[operator]
        ), "The PyTree structure of hists and histsw2 must be the same!"
        self.hists = hists
        self.histsw2 = histsw2
        self.threshold = threshold

        self.ntot = sum_leaves(self.hists)
        self.etot = jnp.sqrt(sum_leaves(self.histsw2))
        ntot_eff = jnp.round(self.ntot**2 / self.etot**2, decimals=0)
        self.mask = ntot_eff > self.threshold

        # setup params
        self.gaussians_global = NormalConstrained(value=jnp.zeros_like(self.ntot))
        self.gaussians_per_process = jtu.tree_map(
            lambda hist: NormalConstrained(value=jnp.zeros_like(hist)), self.hists
        )
        self.poissons_per_process = jtu.tree_map(
            lambda hist: PoissonConstrained(
                lamb=cast(Array, jnp.where(hist > 0.0, hist, 1.0)),
                value=jnp.zeros_like(hist),
            ),
            self.hists,
        )

    def get(self, where: Callable) -> ModifierLike:
        # see: https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/pull/929
        # and: https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/bin-wise-stats/#usage-instructions

        # poisson case per process
        # if w > 0.0, then poisson, else noop (no effect)
        # since w <= 0 leads to NaNs in derivatives, we need to mask them
        w = where(self.hists)
        poisson_params = where(self.poissons_per_process)
        poisson_noop_mod = Modifier(parameter=poisson_params, effect=noop())
        poisson_mod = modifier_where(
            w > 0.0, poisson_params.poisson(), poisson_noop_mod
        )

        # gaussian case per process
        # if w == 0.0, guard for division by zero
        # gaussians with width 0 also lead to nans, so we need to guard this aswell
        w2 = where(self.histsw2)
        relerr = jnp.where(w == 0.0, 0.0, jnp.sqrt(w2) / jnp.where(w == 0.0, 1.0, w))
        mask = relerr == 0.0
        relerr = jnp.where(mask, relerr, 1.0)
        gauss_params = where(self.gaussians_per_process)
        gauss_noop_mod = Modifier(parameter=gauss_params, effect=noop())
        gauss_mod = modifier_where(
            mask, gauss_noop_mod, gauss_params.normal(width=relerr)
        )

        # gaussian case global
        gauss_global_mod = self.gaussians_global.normal(width=self.etot / self.ntot)

        # combine all, logic as here: https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/main/src/CMSHistErrorPropagator.cc#L320-L434
        #
        # legend:
        # - n_tot_eff: effective number of events summed over all processes per bin, `n_tot_eff = n_tot^2 / e_tot^2`
        # - e_tot: error summed over all processes per bin
        # - n_tot: number of events summed over all processes per bin
        # - n_i_eff: effective number of events for process i per bin, `n_i_eff = n_i^2 / e_i^2`
        # - e_i: error for process i per bin
        # - n_i: number of events for process i per bin
        # - threshold: threshold for applying gaussian
        #
        # pseudo-code:
        #
        # if n_tot_eff > threshold:
        #       apply global gaussian(width=e_tot/n_tot)
        # else:
        #   if n_i_eff > threshold or e_i > n_i or n_i <= 0.0:
        #       apply per process gaussian(width=e_i/n_i)
        #   else:
        #       apply per process poisson(lamb=n_i)
        per_process_mask = (
            ((w**2 / w2**2) > self.threshold) | (jnp.sqrt(w2) > w) | (w <= 0)
        )
        return modifier_where(
            self.mask,
            gauss_global_mod,
            modifier_where(per_process_mask, gauss_mod, poisson_mod),
        )
