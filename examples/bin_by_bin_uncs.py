from __future__ import annotations

from operator import itemgetter

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import evermore as evm


class SPlusBModel(eqx.Module):
    staterrors: evm.staterror.StatErrors

    def __init__(self, hists: dict[str, Array], histsw2: dict[str, Array]) -> None:
        # create the staterrors (barlow-beeston-lite with threshold=10.0)
        self.staterrors = evm.staterror.StatErrors(
            hists=hists, histsw2=histsw2, threshold=10.0
        )

    def __call__(self, hists: dict) -> dict[str, Array]:
        expectations = {}

        # signal process
        signal_mcstat_mod = self.staterrors.modifier(getter=lambda p: p["signal"])
        expectations["signal"] = signal_mcstat_mod(hists["signal"])

        # bkg1 process
        bkg1_mcstat_mod = self.staterrors.modifier(getter=lambda p: p["bkg1"])
        expectations["bkg1"] = bkg1_mcstat_mod(hists["bkg1"])

        # bkg2 process
        bkg2_mcstat_mod = self.staterrors.modifier(getter=lambda p: p["bkg2"])
        expectations["bkg2"] = bkg2_mcstat_mod(hists["bkg2"])

        # return the modified expectations
        return expectations


hists = {
    "signal": jnp.array([3.0]),
    "bkg1": jnp.array([10.0]),
    "bkg2": jnp.array([20.0]),
}
histsw2 = {
    "signal": jnp.array([5.0]),
    "bkg1": jnp.array([11.0]),
    "bkg2": jnp.array([25.0]),
}
observation = jnp.array([34.0])

model = SPlusBModel(hists, histsw2)

# test the model
expectations = model(hists)


# scale the histsw2 e.g. after minimization with
# the best fit values of the staterror modifiers.
# This is needed to get the correct stat. uncertainties
modified_histsw2 = {}


for process, histw2 in histsw2.items():
    mod = model.staterrors.modifier(getter=itemgetter(process))
    modified_histsw2[process] = mod(histw2)
