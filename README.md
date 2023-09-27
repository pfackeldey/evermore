# dilax

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

## Usage

```python3
import jax.numpy as jnp

from dilax.likelihood import NLL
from dilax.parameter import Parameter
from dilax.model import Model, EvaluationResult
from dilax.optimizer import JaxOptimizer


class SPlusBModel(Model):
    def evaluate(self) -> EvaluationResult:
        expectations = {}

        expectations["signal"], mu_penalty = self.parameters["mu"](
            self.processes["signal"], type="r"
        )
        expectations["background1"], norm1_penalty = self.parameters["norm1"](
            self.processes["background1"], type="lnN", width=0.1
        )
        expectations["background2"], norm2_penalty = self.parameters["norm2"](
            self.processes["background2"], type="lnN", width=0.05
        )

        penalty = mu_penalty + norm1_penalty + norm2_penalty
        return EvaluationResult(expectations=expectations, penalty=penalty)


def create_model():
    processes = {
        "signal": jnp.array([3]),
        "background1": jnp.array([10]),
        "background2": jnp.array([20]),
    }
    parameters = {
        "mu": Parameter(value=jnp.array([1.0]), bounds=(-jnp.inf, jnp.inf)),
        "norm1": Parameter(value=jnp.array([0.0]), bounds=(-jnp.inf, jnp.inf)),
        "norm2": Parameter(value=jnp.array([0.0]), bounds=(-jnp.inf, jnp.inf)),
    }

    # return model
    return SPlusBModel(processes=processes, parameters=parameters)


model = create_model()

# define data
observation = jnp.array([37])

# create optimizer (from `jaxopt`)
optimizer = JaxOptimizer.make(
    name="LBFGS",
    settings={"maxiter": 5, "jit": True, "unroll": True},
)

# create negative log likelihood
nll = NLL(model=model, observation=observation)

# run a fit
init_values = model.parameter_values
values, state = optimizer.fit(fun=nll, init_values=init_values)

print(values)
>> {'mu': Array([1.1638741], dtype=float32),
 'norm1': Array([0.01125314], dtype=float32),
 'norm2': Array([0.0052684], dtype=float32)}
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
