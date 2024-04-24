import jax.numpy as jnp
from jaxtyping import Array

import evermore as evm

params = {
    "a": evm.Parameter(value=1.0, frozen=True),
    "b": evm.Parameter(value=1.0, prior=evm.pdf.Normal(mean=1.0, width=0.3)),
    "gauss1": {"mu": evm.Parameter(value=0.0), "sigma": evm.Parameter(value=0.5)},
    "gauss2": {"mu": evm.Parameter(value=2.0), "sigma": evm.Parameter(value=0.3)},
}

domain = jnp.arange(0, 5, 100_000)


# model of two stacked gaussians
def model(params, domain):
    def gaussian(x: Array, mu: evm.Parameter, sigma: evm.Parameter) -> Array:
        return jnp.exp(-(((x - mu.value) / sigma.value) ** 2) / 2)

    gauss1 = params["a"].value * gaussian(domain, **params["gauss1"])
    gauss2 = params["b"].value * gaussian(domain, **params["gauss2"])
    return gauss1 + gauss2


# eval model
model(params, domain)
