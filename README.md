<div align="center" style="height:250px;width:250px">
<img src="https://raw.githubusercontent.com/pfackeldey/evermore/main/assets/logo.png" alt="logo"></img>
</div>

# evermore

[![Documentation Status](https://readthedocs.org/projects/evermore/badge/?version=latest)](https://evermore.readthedocs.io/en/latest/?badge=latest)
[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/evermore)](https://github.com/conda-forge/evermore-feedstock)
[![BSD-3 Clause License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

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
from typing import NamedTuple, TypeAlias

from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree, Scalar

import evermore as evm

jax.config.update("jax_enable_x64", True)

# type defs
Hist1D: TypeAlias = Float[Array, "..."]
Hists1D: TypeAlias = dict[str, Hist1D]
Args: TypeAlias = tuple[
    nnx.GraphDef,
    nnx.State,
    Hists1D,
    Hist1D,
]


# define a simple model with two processes and two parameters
def model(params: PyTree, hists: Hists1D) -> Array:
    mu_modifier = params.mu.scale()
    syst_modifier = params.syst.scale_log_asymmetric(up=1.1, down=0.9)
    return mu_modifier(hists["signal"]) + syst_modifier(hists["bkg"])


def loss(
    dynamic: nnx.State,
    args: Args,
) -> Float[Scalar, ""]:
    graphdef, static, hists, observation = args
    params = nnx.merge(graphdef, dynamic, static)
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
graphdef, dynamic, static = nnx.split(params, evm.filter.is_dynamic_parameter, ...)
args = (graphdef, static, hists, observation)

# Calculate negative log-likelihood/loss
loss_val = loss(dynamic, args)
# gradients of negative log-likelihood w.r.t. dynamic parameters
grads = nnx.grad(loss)(dynamic, args)
nnx.display(nnx.pure(grads))
# State({
#   'mu': Array(-0.46153846, dtype=float64, weak_type=True),
#   'syst': Array(-0.15436207, dtype=float64, weak_type=True)
# })
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [BSD license](LICENSE).

<!-- prettier-ignore-start -->

<!-- prettier-ignore-end -->

[actions-badge]: https://github.com/pfackeldey/evermore/workflows/CI/badge.svg
[actions-link]: https://github.com/pfackeldey/evermore/actions
[pypi-link]: https://pypi.org/project/evermore/
[pypi-platforms]: https://img.shields.io/pypi/pyversions/evermore
[pypi-version]: https://img.shields.io/pypi/v/evermore
