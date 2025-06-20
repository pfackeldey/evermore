from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import evermore as evm


class SPlusBModel(eqx.Module):
    staterrors: evm.staterror.StatErrors

    def __init__(self, hists: dict[str, Array], histsw2: dict[str, Array]) -> None:
        # create the staterrors (barlow-beeston-full -> per-process)
        self.staterrors = jax.tree.map(
            evm.staterror.StatErrors,
            hists,
            histsw2,
        )

    def __call__(self, hists: dict) -> dict[str, Array]:
        expectations = {}

        # signal process
        expectations["signal"] = self.staterrors["signal"](hists["signal"])

        # bkg1 process
        expectations["bkg1"] = self.staterrors["bkg1"](hists["bkg1"])

        # bkg2 process
        expectations["bkg2"] = self.staterrors["bkg2"](hists["bkg2"])

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
