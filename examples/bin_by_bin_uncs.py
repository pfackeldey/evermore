from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, PyTree

import evermore as evm

jax.config.update("jax_enable_x64", True)

Hist1D = tp.TypeVar("Hist1D", bound=Float[Array, " nbins"])


class ModelWithStatErrors(nnx.Module):
    def __init__(self, hists: PyTree[Hist1D], variances: PyTree[Hist1D]) -> None:
        self.staterrors = nnx.Dict(
            jax.tree.map(evm.staterror.StatErrors, hists, histsw2)
        )

    def __call__(self, hists: PyTree[Hist1D]) -> PyTree[Hist1D]:
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


if __name__ == "__main__":
    # test the model
    model_with_staterrors = ModelWithStatErrors(hists, histsw2)
    expectations = model_with_staterrors(hists)
    print(expectations)
