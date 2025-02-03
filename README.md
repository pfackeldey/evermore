<div align="center" style="height:250px;width:250px">
<img src="https://raw.githubusercontent.com/pfackeldey/evermore/main/assets/logo.png" alt="logo"></img>
</div>

# evermore

[![Documentation Status](https://readthedocs.org/projects/evermore/badge/?version=latest)](https://evermore.readthedocs.io/en/latest/?badge=latest)
[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

Differentiable (binned) likelihoods in JAX.

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

## Example - Model and Loss Definition

See more in `examples/`

_evermore_ in a nutshell:

```python3
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
    log_likelihood = evm.pdf.Poisson(lamb=expectation).log_prob(observation).sum()
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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [BSD license](LICENSE).

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/pfackeldey/evermore/workflows/CI/badge.svg
[actions-link]:             https://github.com/pfackeldey/evermore/actions
[pypi-link]:                https://pypi.org/project/evermore/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/evermore
[pypi-version]:             https://img.shields.io/pypi/v/evermore
<!-- prettier-ignore-end -->
