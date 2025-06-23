from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

import evermore as evm

jax.config.update("jax_enable_x64", True)


# dataclass like container for parameters
class Params(eqx.Module):
    mu: evm.Parameter
    norm1: evm.NormalParameter
    norm2: evm.NormalParameter
    shape1: evm.NormalParameter


def model(
    params: PyTree[evm.Parameter],
    hists: PyTree[Float[Array, " nbins"]],
) -> PyTree[Float[Array, " nbins"]]:
    expectations = {}

    # signal process
    sig_mod = params.mu.scale()
    expectations["signal"] = sig_mod(hists["nominal"]["signal"])

    # bkg1 process
    bkg1_lnN = params.norm1.scale_log(up=jnp.array([1.1]), down=jnp.array([0.9]))
    bkg1_shape = params.shape1.morphing(
        up_template=hists["shape_up"]["bkg1"],
        down_template=hists["shape_down"]["bkg1"],
    )
    # combine modifiers
    bkg1_mod = bkg1_lnN @ bkg1_shape
    expectations["bkg1"] = bkg1_mod(hists["nominal"]["bkg1"])

    # bkg2 process
    bkg2_lnN = params.norm2.scale_log(up=jnp.array([1.05]), down=jnp.array([0.95]))
    bkg2_shape = params.shape1.morphing(
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
        "signal": jnp.array([3.0]),
        "bkg1": jnp.array([10.0]),
        "bkg2": jnp.array([20.0]),
    },
    "shape_up": {
        "bkg1": jnp.array([12.0]),
        "bkg2": jnp.array([23.0]),
    },
    "shape_down": {
        "bkg1": jnp.array([8.0]),
        "bkg2": jnp.array([19.0]),
    },
}

params = Params(
    mu=evm.Parameter(),  # type: ignore[arg-type]
    norm1=evm.NormalParameter(),
    norm2=evm.NormalParameter(),
    shape1=evm.NormalParameter(),
)

observation = jnp.array([37.0])
expectations = model(params, hists)
