from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import evermore as evm


class SPlusBModel(eqx.Module):
    mu: evm.Parameter
    norm1: evm.NormalParameter
    norm2: evm.NormalParameter
    shape1: evm.NormalParameter

    def __call__(self, hists: dict) -> dict[str, Array]:
        expectations = {}

        # signal process
        sig_mod = self.mu.scale()
        expectations["signal"] = sig_mod(hists["nominal"]["signal"])

        # bkg1 process
        bkg1_lnN = self.norm1.scale_log(up=jnp.array([1.1]), down=jnp.array([0.9]))
        bkg1_shape = self.shape1.morphing(
            up_template=hists["shape_up"]["bkg1"],
            down_template=hists["shape_down"]["bkg1"],
        )
        # combine modifiers
        bkg1_mod = bkg1_lnN @ bkg1_shape
        expectations["bkg1"] = bkg1_mod(hists["nominal"]["bkg1"])

        # bkg2 process
        bkg2_lnN = self.norm2.scale_log(up=jnp.array([1.05]), down=jnp.array([0.95]))
        bkg2_shape = self.shape1.morphing(
            up_template=hists["shape_up"]["bkg2"],
            down_template=hists["shape_down"]["bkg2"],
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

model = SPlusBModel(
    mu=evm.Parameter(value=0.0, lower=0.0, upper=10.0),
    norm1=evm.NormalParameter(),
    norm2=evm.NormalParameter(),
    shape1=evm.NormalParameter(),
)

observation = jnp.array([37])
expectations = model(hists)
