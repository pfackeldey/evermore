from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from evermore.custom_types import ModifierLike
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

            # Create a modifier for the qcd process, `get` is a function
            # that finds the corresponding parameter from `staterrors.params_per_process`
            mod = staterrors(get=lambda x: x["qcd"])
            # apply the modifier to the parameter
            mod(hists["qcd"])
    """

    params_global: PyTree
    params_per_process: PyTree
    etot: Array = eqx.field(static=True)
    mask: Array = eqx.field(static=True)

    def __init__(
        self,
        hists: PyTree,
        histsw2: PyTree,
        threshold: float = 10.0,
    ) -> None:
        leaf = jtu.tree_leaves(hists)[0]
        self.params_global = NormalConstrained(value=jnp.zeros_like(leaf))
        self.params_per_process = jtu.tree_map(
            lambda hist: PoissonConstrained(lamb=hist), hists
        )
        wtot = sum_leaves(hists)
        self.etot = jnp.sqrt(sum_leaves(histsw2))
        wtot_eff = jnp.round(wtot**2 / self.etot**2, decimals=0)
        self.mask = wtot_eff > threshold

    def get(self, where: Callable) -> ModifierLike:
        poisson_mod = where(self.params_per_process).poisson()
        normal_mod = self.params_global.normal(width=self.etot)
        return modifier_where(self.mask, normal_mod, poisson_mod)
