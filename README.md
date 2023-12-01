# dilax

[![Documentation Status](https://readthedocs.org/projects/dilax/badge/?version=latest)](https://dilax.readthedocs.io/en/latest/?badge=latest)
[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

Differentiable (binned) likelihoods in JAX.

## Installation

```bash
python -m pip install dilax
```

From source:

```bash
git clone https://github.com/pfackeldey/dilax
cd dilax
python -m pip install .
```

## Usage - Model definition and fitting

See more in `examples/`

_dilax_ in a nutshell:

```python3
import equinox as eqx
import jax
import jax.numpy as jnp

import dilax as dlx

jax.config.update("jax_enable_x64", True)


# define a simple model with two processes and two parameters
class MyModel(dlx.Model):
    def __call__(self, processes: dict, parameters: dict) -> dlx.Result:
        res = dlx.Result()

        # signal
        mu_mod = dlx.modifier(
            name="mu", parameter=parameters["mu"], effect=dlx.effect.unconstrained()
        )
        res.add(process="signal", expectation=mu_mod(processes["signal"]))

        # background
        bkg_mod = dlx.modifier(
            name="sigma", parameter=parameters["sigma"], effect=dlx.effect.gauss(0.2)
        )
        res.add(process="background", expectation=bkg_mod(processes["background"]))
        return res


# setup model
processes = {"signal": jnp.array([10.0]), "background": jnp.array([50.0])}
parameters = {
    "mu": dlx.Parameter(value=jnp.array([1.0]), bounds=(0.0, jnp.inf)),
    "sigma": dlx.Parameter(value=jnp.array([0.0])),
}
model = MyModel(processes=processes, parameters=parameters)

# define negative log-likelihood with data (observation)
nll = dlx.likelihood.NLL(model=model, observation=jnp.array([64.0]))
# jit it!
fast_nll = eqx.filter_jit(nll)

# setup fit: initial values of parameters and a suitable optimizer
init_values = model.parameter_values
optimizer = dlx.optimizer.JaxOptimizer.make(
    name="ScipyMinimize", settings={"method": "trust-constr"}
)

# fit
values, state = optimizer.fit(fun=fast_nll, init_values=init_values)

print(values)
# -> {'mu': Array([1.4], dtype=float64),
#     'sigma': Array([4.04723836e-14], dtype=float64)}

# eval model with fitted values
print(model.update(values=values).evaluate().expectation())
# -> Array([64.], dtype=float64)


# gradients of "prefit" model:
print(eqx.filter_grad(nll)({"sigma": jnp.array([0.2])}))
# -> {'sigma': Array([-0.12258065], dtype=float64)}


# gradients of "postfit" model:
@eqx.filter_grad
@eqx.filter_jit
def grad_postfit_nll(where: dict[str, jax.Array]) -> dict[str, jax.Array]:
    nll = dlx.likelihood.NLL(
        model=model.update(values=values), observation=jnp.array([64.0])
    )
    return nll(where)


print(grad_postfit_nll({"sigma": jnp.array([0.2])}))
# -> {'sigma': Array([0.5030303], dtype=float64)}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [BSD license](LICENSE).

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/pfackeldey/dilax/workflows/CI/badge.svg
[actions-link]:             https://github.com/pfackeldey/dilax/actions
[pypi-link]:                https://pypi.org/project/dilax/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/dilax
[pypi-version]:             https://img.shields.io/pypi/v/dilax
<!-- prettier-ignore-end -->
