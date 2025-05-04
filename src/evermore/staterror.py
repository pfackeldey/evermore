from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from evermore.custom_types import ModifierLike
from evermore.effect import Identity
from evermore.modifier import Modifier, Where
from evermore.parameter import NormalParameter
from evermore.util import sum_over_leaves
from evermore.visualization import SupportsTreescope

__all__ = [
    "StatErrors",
]


def __dir__():
    return __all__


class StatErrors(eqx.Module, SupportsTreescope):
    """
    Create staterror (barlow-beeston) parameters.

    Example:

    .. code-block:: python

        from operator import itemgetter
        import jax.numpy as jnp
        import evermore as evm


        # histograms with bin contents and variances
        hists = {"qcd": jnp.array([10, 20, 30]), "signal": jnp.array([5, 10, 15])}
        histsw2 = {"qcd": jnp.array([12, 21, 29]), "signal": jnp.array([5, 8, 11])}

        # threshold deciding above which number of true entries a global gaussian
        # parameter is used per bin rather than per-histogram ones
        # (negative values mean that per-histogram gaussians are always used)
        threshold = -1.0

        staterrors = evm.staterror.StatErrors.from_hists_and_variances(hists, histsw2, treshold)

        # Create a modifier for the qcd process, `getter` is a function
        # that finds the corresponding parameter from for the Poissons and Gaussians
        getter = itemgetter("qcd")
        mod = staterrors.modifier(getter=getter)
        # apply the modifier to the parameter
        mod(getter(hists))
    """

    n_entries: PyTree[Array]
    gaussians_per_hist: PyTree
    gaussians_global: PyTree
    global_mask: Array
    threshold: float

    @classmethod
    def from_hists_and_variances(
        cls,
        hists: PyTree[Array],
        variances: PyTree[Array],
        threshold: float = -1.0,
    ) -> StatErrors:
        assert (
            jax.tree.structure(hists) == jax.tree.structure(variances)  # type: ignore[operator]
        ), "The PyTree structure of hists and variances must be the same!"
        n_entries = jax.tree.map(
            lambda w, w2: jnp.where(w2 != 0.0, (w**2 / (w2 + jnp.where(w2 != 0.0, 0.0, jnp.finfo(w2.dtype).eps))), 0.0),
            hists,
            variances,
        )
        return cls(n_entries=n_entries, threshold=threshold)

    def __init__(self, n_entries: PyTree[Array], threshold: float = -1.0) -> None:
        self.n_entries = n_entries
        self.threshold = threshold

        # setup gaussian per-hist params
        self.gaussians_per_hist = jax.tree.map(lambda n: NormalParameter(value=jnp.zeros_like(n)), self.n_entries)

        # setup gaussian global params
        n_tot_entries = sum_over_leaves(self.n_entries)
        global_zeros = jnp.zeros_like(n_tot_entries)
        self.gaussians_global = NormalParameter(value=global_zeros)

        # evaluate the mask for switching between per-hist and global params
        self.global_mask = jnp.astype(global_zeros, jnp.bool) if threshold < 0.0 else n_tot_entries >= threshold

    def modifier(self, getter: Callable) -> ModifierLike:
        """
        Creates a modifier for statistical errors (Barlow-Beeston parameters)
        for a given histogram. This modifier applies a Gaussian treatment to the
        statistical uncertainties based on the number of true entries. Depending
        on the threshold, the modifier will apply either a global Gaussian per bin
        or per-histogram Gaussians.

        Args:
            getter (Callable): A function to extract the relevant histogram
                and variance for a specific hist.

        Returns:
            ModifierLike: A modifier that applies the appropriate statistical
                Gaussian treatment based on the input data.
        """
        # see: https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/pull/929
        # and: https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/bin-wise-stats/#usage-instructions
        # and: https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/pull/929#issuecomment-2034000873
        # however, note that the treatment is implemented as always being Gaussian below

        # extract entries
        n = getter(self.n_entries)
        n_tot = sum_over_leaves(self.n_entries)
        eps = jnp.finfo(n.dtype).eps

        # gaussian per-hist case
        non_empty_mask = n != 0.0
        rel_err = jnp.where(
            non_empty_mask,
            1.0 / jnp.sqrt(n + jnp.where(non_empty_mask, 0.0, eps)),
            1.0,
        )
        gauss_hist_params = getter(self.gaussians_per_hist)
        per_hist_mod = Where(
            non_empty_mask,
            gauss_hist_params.scale(slope=rel_err, offset=1.0),
            Modifier(parameter=gauss_hist_params, effect=Identity()),
        )

        # gaussian global case
        global_non_empty_mask = n_tot != 0.0
        rel_err_tot = jnp.where(
            global_non_empty_mask,
            1.0 / jnp.sqrt(n_tot + jnp.where(global_non_empty_mask, 0.0, eps)),
            1.0,
        )
        global_mod = Where(
            global_non_empty_mask,
            self.gaussians_global.scale(slope=rel_err_tot, offset=1.0),
            Modifier(parameter=self.gaussians_global, effect=Identity()),
        )

        # combine both
        return Where(self.global_mask, global_mod, per_hist_mod)
