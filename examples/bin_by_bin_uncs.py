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
        self.staterrors = evm.staterror.StatErrors.from_hists_and_variances(
            hists=hists, variances=histsw2
        )

    def __call__(self, hists: dict) -> dict[str, Array]:
        expectations = {}

        # signal process
        getter = itemgetter("signal")
        signal_mcstat_mod = self.staterrors.modifier(getter=getter)
        expectations["signal"] = signal_mcstat_mod(getter(hists))

        # bkg1 process
        getter = itemgetter("bkg1")
        bkg1_mcstat_mod = self.staterrors.modifier(getter=getter)
        expectations["bkg1"] = bkg1_mcstat_mod(getter(hists))

        # bkg2 process
        getter = itemgetter("bkg2")
        bkg2_mcstat_mod = self.staterrors.modifier(getter=getter)
        expectations["bkg2"] = bkg2_mcstat_mod(getter(hists))

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
