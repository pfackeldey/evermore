from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import evermore as evm


class SPlusBModel(eqx.Module):
    mu: evm.Parameter
    norm1: evm.Parameter
    norm2: evm.Parameter
    shape1: evm.Parameter

    def __init__(self, hist: dict[str, Array], histw2: dict[str, Array]) -> None:
        self.mu = evm.Parameter(value=jnp.array([1.0]))
        self = evm.parameter.auto_init(self)

    def __call__(self, hists: dict) -> dict[str, Array]:
        expectations = {}

        # signal process
        sig_mod = self.mu.unconstrained()
        expectations["signal"] = sig_mod(hists["nominal"]["signal"])

        # bkg1 process
        bkg1_lnN = self.norm1.lnN(up=jnp.array([1.1]), down=jnp.array([0.9]))
        bkg1_shape = self.shape1.shape(
            up=hists["shape_up"]["bkg1"],
            down=hists["shape_down"]["bkg1"],
        )
        # combine modifiers
        bkg1_mod = bkg1_lnN @ bkg1_shape
        expectations["bkg1"] = bkg1_mod(hists["nominal"]["bkg1"])

        # bkg2 process
        bkg2_lnN = self.norm2.lnN(up=jnp.array([1.05]), down=jnp.array([0.95]))
        bkg2_shape = self.shape1.shape(
            up=hists["shape_up"]["bkg2"],
            down=hists["shape_down"]["bkg2"],
        )
        # combine modifiers
        bkg2_mod = bkg2_lnN @ bkg2_shape
        expectations["bkg2"] = bkg2_mod(hists["nominal"]["bkg2"])

        # return the modified expectations
        return expectations


hists = {
    "nominal": {
        "signal": jnp.array([3]),
        "bkg1": jnp.array([10]),
        "bkg2": jnp.array([20]),
    },
    "shape_up": {
        "bkg1": jnp.array([12]),
        "bkg2": jnp.array([23]),
    },
    "shape_down": {
        "bkg1": jnp.array([8]),
        "bkg2": jnp.array([19]),
    },
}

hist = hists["nominal"]
histw2 = {
    "signal": jnp.array([5]),
    "bkg1": jnp.array([11]),
    "bkg2": jnp.array([25]),
}

model = SPlusBModel(hist, histw2)

observation = jnp.array([37])
expectations = model(hists)
