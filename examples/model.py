from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

import evermore as evm


class SPlusBModel(eqx.Module):
    mu: evm.Parameter
    norm1: evm.Parameter
    norm2: evm.Parameter
    shape1: evm.Parameter

    def __init__(self) -> None:
        self.mu = evm.Parameter(value=jnp.array([1.0]))
        self = evm.parameter.auto_init(self)

    def __call__(self, hists: dict[Any, jax.Array]) -> dict[str, jax.Array]:
        expectations = {}

        # signal process
        sig_mod = self.mu.unconstrained()
        expectations["signal"] = sig_mod(hists[("signal", "nominal")])

        # bkg1 process
        bkg1_mod = self.norm1.lnN(width=jnp.array([0.9, 1.1])) @ self.shape1.shape(
            up=hists[("bkg1", "shape_up")],
            down=hists[("bkg1", "shape_down")],
        )
        expectations["bkg1"] = bkg1_mod(hists[("bkg1", "nominal")])

        # bkg2 process
        bkg2_mod = self.norm2.lnN(width=jnp.array([0.95, 1.05])) @ self.shape1.shape(
            up=hists[("bkg2", "shape_up")],
            down=hists[("bkg2", "shape_down")],
        )
        expectations["bkg2"] = bkg2_mod(hists[("bkg2", "nominal")])

        # return the modified expectations
        return expectations


model = SPlusBModel()


hists = {
    ("signal", "nominal"): jnp.array([3]),
    ("bkg1", "nominal"): jnp.array([10]),
    ("bkg2", "nominal"): jnp.array([20]),
    ("bkg1", "shape_up"): jnp.array([12]),
    ("bkg1", "shape_down"): jnp.array([8]),
    ("bkg2", "shape_up"): jnp.array([23]),
    ("bkg2", "shape_down"): jnp.array([19]),
}

observation = jnp.array([37])
expectations = model(hists)
