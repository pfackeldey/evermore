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
from typing import NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import wadler_lindig as wl
from jaxtyping import Array, Float, Scalar

import evermore as evm

jax.config.update("jax_enable_x64", True)

Hist1D: TypeAlias = Float[Array, "..."]
Hists1D: TypeAlias = dict[str, Hist1D]


# define a simple model with two processes and two parameters
def model(params: evm.PT, hists: Hists1D) -> Array:
    mu_modifier = params.mu.scale()
    syst_modifier = params.syst.scale_log(up=1.1, down=0.9)
    return mu_modifier(hists["signal"]) + syst_modifier(hists["bkg"])


def loss(
    dynamic: evm.PT,
    static: evm.PT,
    hists: Hists1D,
    observation: Hist1D,
) -> Float[Scalar, ""]:
    params = evm.tree.combine(dynamic, static)
    expectation = model(params, hists)
    # Poisson NLL of the expectation and observation
    log_likelihood = (
        evm.pdf.PoissonContinuous(lamb=expectation).log_prob(observation).sum()
    )
    # Add parameter constraints from logpdfs
    constraints = evm.loss.get_log_probs(params)
    log_likelihood += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(log_likelihood)


# setup data
hists: Hists1D = {"signal": jnp.array([3.0]), "bkg": jnp.array([10.0])}
observation: Hist1D = jnp.array([15.0])


# define parameters, can be any PyTree of evm.Parameters
class Params(NamedTuple):
    mu: evm.Parameter[Float[Scalar, ""]]
    syst: evm.NormalParameter[Float[Scalar, ""]]


params = Params(mu=evm.Parameter(1.0), syst=evm.NormalParameter(0.0))

# split tree of parameters in a differentiable part and a static part
dynamic, static = evm.tree.partition(params)

# Calculate negative log-likelihood/loss
loss_val = loss(dynamic, static, hists, observation)
# gradients of negative log-likelihood w.r.t. dynamic parameters
grads = eqx.filter_grad(loss)(dynamic, static, hists, observation)
wl.pprint(evm.tree.pure(grads), short_arrays=False)
# -> Params(mu=Array(-0.46153846, dtype=float64), syst=Array(-0.15436207, dtype=float64))

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
