# Getting Started

evermore is a toolbox that provides common building blocks for building (binned) likelihoods in high-energy physics with JAX.

## Installation

```bash
python -m pip install evermore
```

From source:

```bash
git clone https://github.com/pfackeldey/evermore
cd evermore
python -m pip install .
```


## evermore Quickstart
```{code-block} python
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

import evermore as evm

jax.config.update("jax_enable_x64", True)


# define a simple model with two processes and two parameters
def model(params: PyTree, hists: dict[str, Array]) -> Array:
  mu_modifier = params.mu.scale()
  syst_modifier = params.syst.scale_log(up=1.1, down=0.9)
  return mu_modifier(hists["signal"]) + syst_modifier(hists["bkg"])


def loss(
  diffable: PyTree,
  static: PyTree,
  hists: dict[str, Array],
  observation: Array,
) -> Array:
    params = eqx.combine(diffable, static)
    expectation = model(params, hists)
    # Poisson NLL of the expectation and observation
    log_likelihood = evm.loss.PoissonLogLikelihood()(expectation, observation)
    # Add parameter constraints from logpdfs
    constraints = evm.loss.get_log_probs(params)
    log_likelihood += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(log_likelihood)


# setup data
hists = {"signal": jnp.array([3]), "bkg": jnp.array([10])}
observation = jnp.array([15])


# define parameters, can be any PyTree of evm.Parameters
class Params(NamedTuple):
  mu: evm.Parameter
  syst: evm.NormalParameter


params = Params(mu=evm.Parameter(1.0), syst=evm.NormalParameter(0.0))
diffable, static = evm.parameter.partition(params)

# Calculate negative log-likelihood/loss
loss_val = loss(diffable, static, hists, observation)
# gradients of negative log-likelihood w.r.t. diffable parameters
grads = eqx.filter_grad(loss)(diffable, static, hists, observation)
print(f"{grads.mu.value=}, {grads.syst.value=}")
# -> grads.mu.value=Array([-0.46153846]), grads.syst.value=Array([-0.15436207])
```

Checkout the other [Examples](https://github.com/pfackeldey/evermore/tree/main/examples).

## Table of Contents
```{toctree}
:maxdepth: 2
binned_likelihood.md
building_blocks.md
tips_and_tricks.md
evermore_for_CMS.md
evermore_for_ATLAS.md
api/index.md
```
