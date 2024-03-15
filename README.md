<div align="center" style="height:250px;width:250px">
<img src="https://raw.githubusercontent.com/pfackeldey/evermore/main/assets/logo.png" alt="logo"></img>
</div>

# evermore

[![Documentation Status](https://readthedocs.org/projects/dilax/badge/?version=latest)](https://dilax.readthedocs.io/en/latest/?badge=latest)
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
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import evermore as evm

jax.config.update("jax_enable_x64", True)


# define a simple model with two processes and two parameters
class Model(eqx.Module):
    mu: evm.Parameter
    syst: evm.Parameter

    def __call__(self, hists: dict[str, Array]) -> Array:
        mu_modifier = self.mu.unconstrained()
        syst_modifier = self.syst.lnN(up=jnp.array([1.1]), down=jnp.array([0.9]))
        return mu_modifier(hists["signal"]) + syst_modifier(hists["bkg"])


nll = evm.loss.PoissonNLL()


def loss(model: Model, hists: dict[str, Array], observation: Array) -> Array:
    expectation = model(hists)
    # Poisson NLL of the expectation and observation
    log_likelihood = nll(expectation, observation)
    # Add parameter constraints from logpdfs
    constraints = evm.loss.get_param_constraints(model)
    log_likelihood += evm.util.sum_leaves(constraints)
    return -jnp.sum(log_likelihood)


# setup model and data
hists = {"signal": jnp.array([3]), "bkg": jnp.array([10])}
observation = jnp.array([15])
model = Model(mu=evm.Parameter(1.0), syst=evm.Parameter(0.0))

# negative log-likelihood
loss_val = loss(model, hists, observation)
# gradients of negative log-likelihood w.r.t. model parameters
grads = eqx.filter_grad(loss)(model, hists, observation)
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
