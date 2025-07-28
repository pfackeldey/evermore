from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

import evermore as evm

jax.config.update("jax_enable_x64", True)

Hist1D = tp.TypeVar("Hist1D", bound=Float[Array, " nbins"])


def model(
    staterrors: evm.staterror.StatErrors, hists: PyTree[Hist1D]
) -> PyTree[Hist1D]:
    expectations = {}

    # signal process
    expectations["signal"] = staterrors["signal"](hists["signal"])

    # bkg1 process
    expectations["bkg1"] = staterrors["bkg1"](hists["bkg1"])

    # bkg2 process
    expectations["bkg2"] = staterrors["bkg2"](hists["bkg2"])

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

# `staterrors` is just a pytree of `evm.Parameter`(s)
staterrors = jax.tree.map(
    evm.staterror.StatErrors,
    hists,
    histsw2,
)

if __name__ == "__main__":
    # test the model
    expectations = model(staterrors, hists)
