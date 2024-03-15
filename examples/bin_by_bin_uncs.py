from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

import evermore as evm


class SPlusBModel(eqx.Module):
    staterrors: PyTree[evm.Parameter]

    def __init__(self, hists: dict[str, Array]) -> None:
        self.staterrors = evm.parameter.staterrors(hists=hists)

    def __call__(self, hists: dict, histsw2: dict) -> dict[str, Array]:
        expectations = {}

        # calculate widths of the sum of the nominal histograms for gaussian MC stat
        sqrtw2 = jtu.tree_map(jnp.sqrt, histsw2)
        widths = evm.util.sum_leaves(sqrtw2) / evm.util.sum_leaves(hists)
        gauss_mcstat = self.staterrors["gauss"].gauss(widths)
        # barlow-beeston-like condition: above 10 use gauss, below use poisson
        mask = evm.util.sum_leaves(hists) > 10

        # signal process
        signal_poisson = self.staterrors["poisson"]["signal"].poisson(hists["signal"])
        signal_mc_stats = evm.modifier.where(mask, gauss_mcstat, signal_poisson)
        expectations["signal"] = signal_mc_stats(hists["signal"])

        # bkg1 process
        bkg1_poisson = self.staterrors["poisson"]["bkg1"].poisson(hists["bkg1"])
        bkg1_mc_stats = evm.modifier.where(mask, gauss_mcstat, bkg1_poisson)
        expectations["bkg1"] = bkg1_mc_stats(hists["bkg1"])

        # bkg2 process
        bkg2_poisson = self.staterrors["poisson"]["bkg2"].poisson(hists["bkg2"])
        bkg2_mc_stats = evm.modifier.where(mask, gauss_mcstat, bkg2_poisson)
        expectations["bkg2"] = bkg2_mc_stats(hists["bkg2"])

        # return the modified expectations
        return expectations


hists = {
    "signal": jnp.array([3]),
    "bkg1": jnp.array([10]),
    "bkg2": jnp.array([20]),
}
histsw2 = {
    "signal": jnp.array([5]),
    "bkg1": jnp.array([11]),
    "bkg2": jnp.array([25]),
}

model = SPlusBModel(hists)

# test the model
expectations = model(hists, histsw2)
