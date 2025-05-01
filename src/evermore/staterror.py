from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from evermore.custom_types import ModifierLike
from evermore.effect import Identity
from evermore.modifier import Modifier, Where
from evermore.parameter import NormalParameter, Parameter
from evermore.pdf import Poisson
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


        hists = {"qcd": jnp.array([10, 20, 30]), "signal": jnp.array([5, 10, 15])}
        histsw2 = {"qcd": jnp.array([12, 21, 29]), "signal": jnp.array([5, 8, 11])}

        staterrors = evm.staterror.StatErrors.from_hists_and_variances(hists, histsw2)

        # Create a modifier for the qcd process, `getter` is a function
        # that finds the corresponding parameter from for the Poissons and Gaussians
        getter = itemgetter("qcd")
        mod = staterrors.modifier(getter=getter, hist=getter(hists))
        # apply the modifier to the parameter
        mod(getter(hists))
    """

    gaussians_per_process: PyTree
    poissons_per_process: PyTree
    n_true_events: PyTree[Array]

    @classmethod
    def from_hists_and_variances(cls, hists: PyTree[Array], variances: PyTree[Array]):
        assert (
            jax.tree.structure(hists) == jax.tree.structure(variances)  # type: ignore[operator]
        ), "The PyTree structure of hists and variances must be the same!"
        n_true_events = jax.tree.map(
            lambda w, w2: jnp.where(w != 0.0, (w**2 / w2), 0.0),
            hists,
            variances,
        )
        return cls(n_true_events=n_true_events)

    def __init__(self, n_true_events: PyTree[Array]) -> None:
        self.n_true_events = n_true_events

        # setup params
        self.gaussians_per_process = jax.tree.map(
            lambda n: NormalParameter(value=jnp.zeros_like(n)), self.n_true_events
        )
        self.poissons_per_process = jax.tree.map(
            lambda n: Parameter(
                value=jnp.zeros_like(n),
                prior=Poisson(lamb=jnp.where(n != 0.0, n, 0.0)),
            ),
            self.n_true_events,
        )

    def modifier(
        self,
        getter: Callable,
        hist: Array,
    ) -> ModifierLike:
        """
        Creates a modifier for statistical errors (Barlow-Beeston parameters)
        for a given process. This modifier applies either a Poisson or Gaussian
        treatment to the statistical uncertainties based on the input histograms
        and their associated uncertainties.

        Args:
            getter (Callable): A function to extract the relevant histogram
                and variance for a specific process.
            hist (Array): Histogram values for the process.

        Returns:
            ModifierLike: A modifier that applies the appropriate statistical
            treatment (Poisson or Gaussian) based on the input data.
        """
        # see: https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/pull/929
        # and: https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/bin-wise-stats/#usage-instructions
        # and: https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/pull/929#issuecomment-2034000873
        # however, note that the statistical simplification of treating the sum of poissons as a
        # single gaussian per bin is not applied here, as the full treatment is encouraged

        # poisson case per process
        # if n != 0.0, then poisson, else Identity (no effect)
        n = getter(self.n_true_events)
        non_empty_mask = n != 0.0
        poisson_params = getter(self.poissons_per_process)
        poisson_identity_mod = Modifier(parameter=poisson_params, effect=Identity())
        poisson_mod = Where(
            non_empty_mask,
            poisson_params.scale(slope=1.0, offset=1.0),
            poisson_identity_mod,
        )

        # gaussian case per process
        eps = jnp.finfo(n.dtype).eps
        relerr = jnp.where(
            non_empty_mask,
            1.0 / jnp.sqrt(n + jnp.where(non_empty_mask, 0.0, eps)),
            1.0,
        )
        gauss_params = getter(self.gaussians_per_process)
        gauss_identity_mod = Modifier(parameter=gauss_params, effect=Identity())
        gauss_mod = Where(
            non_empty_mask,
            gauss_params.scale(slope=relerr, offset=1.0),
            gauss_identity_mod,
        )

        # combine both, logic as here: https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/main/src/CMSHistErrorPropagator.cc#L320-L434
        #
        # legend:
        # - n_i: number of events for process i per bin
        # - n_i_true: true number of events for process i per bin, `n_i_true ~= n_i^2 / e_i^2`
        #
        # pseudo-code (per bin):
        #
        # if n_i_true < 1 or n_i <= 0.0:
        #     apply per process gaussian(width=1/sqrt(n_i_true))
        # else:
        #     apply per process poisson(lamb=n_i_true)
        gauss_mask = (n < 1.0) | (hist <= 0.0)
        return Where(gauss_mask, gauss_mod, poisson_mod)
