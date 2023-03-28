# dilax
differentiable (binned) likelihoods with JAX


## TL;DR

```python
import jax
import jax.numpy as jnp

import chex

from dilax.likelihood import nll
from dilax.parameter import r, lnN
from dilax.model import Model
from dilax.optimizer import JaxOptimizer


# Define model; i.e. how is the expectation calculated with (nuisance) parameters?
@chex.dataclass(frozen=True)
class SPlusBModel(Model):
    @jax.jit
    def eval(self) -> jax.Array:
        expectation = jnp.array(0.0)

        # modify affected processes
        for process, sumw in self.processes.items():
            # mu
            if process == "signal":
                sumw = self.parameters["mu"].apply(sumw)

            # background norm
            elif process == "bkg":
                sumw = self.parameters["norm"].apply(sumw)

            expectation += sumw
        return expectation


# Initialize S+B model
model = SPlusBModel(
    processes={"signal": jnp.array([3.0]), "bkg": jnp.array([10.0])},
    parameters={"mu": r(strength=jnp.array(1.0)), "norm": lnN(strength=jnp.array(0.0), width=jnp.array(0.1))},
)

# Define data
observation = jnp.array([15.0])

# Setup optimizer, see more at https://jaxopt.github.io/stable/ 
optimizer = JaxOptimizer.make(name="LBFGS", settings={"maxiter": 30, "tol": 1e-6})

# Run fit
params, state = optimizer.fit(fun=nll, init_params=model.parameter_strengths, model=model, observation=observation)

print(params)
>> {'mu': DeviceArray(1.6666667, dtype=float32), 'norm': DeviceArray(0., dtype=float32)}
```

### See more in `examples/`
