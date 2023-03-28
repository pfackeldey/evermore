import jax
import jax.numpy as jnp

import chex

from dilax.parameter import r, lnN, add_mc_stats
from dilax.model import Model
from dilax.optimizer import JaxOptimizer


@chex.dataclass(frozen=True)
class SPlusBModel(Model):
    @jax.jit
    def eval(self) -> jax.Array:
        expectation = jnp.array(0.0)

        # modify affected processes
        for process, (sumw, sumw2) in self.processes.items():
            # mu
            if process == "signal":
                sumw = self.parameters["mu"].apply(sumw)

            # background norms
            elif process == "background1":
                sumw = self.parameters["norm1"].apply(sumw)
            elif process == "background2":
                sumw = self.parameters["norm2"].apply(sumw)

            # mc stat per bin per process
            for i in range(sumw.shape[0]):
                mcstat = f"mcstat_{process}_{i}"
                if mcstat in self.parameters:
                    sumw = sumw.at[i].set(self.parameters[mcstat].apply(sumw[i]))

            expectation += sumw

        # mc stat per bin
        mcstat = f"mcstat_{i}"
        for i in range(expectation.shape[0]):
            if mcstat in self.parameters:
                expectation = expectation.at[i].set(self.parameters[mcstat].apply(expectation[i]))

        return expectation


def create_model():
    processes = {
        "signal": (jnp.array([3]), jnp.array([4])),
        "background1": (jnp.array([10]), jnp.array([12])),
        "background2": (jnp.array([20]), jnp.array([30])),
    }
    parameters = {
        "mu": r(strength=jnp.array(1.0)),
        "norm1": lnN(strength=jnp.array(0.0), width=jnp.array(0.1)),
        "norm2": lnN(strength=jnp.array(0.0), width=jnp.array(0.05)),
    }

    parameters.update(add_mc_stats(processes=processes, treshold=10, prefix="mcstat"))

    # return model
    return SPlusBModel(processes=processes, parameters=parameters)


model = create_model()


init_params = model.parameter_strengths
observation = jnp.array([37])


# create optimizer (from `jaxopt`)
optimizer = JaxOptimizer.make(
    name="LBFGS",
    settings={
        "maxiter": 30,
        "tol": 1e-6,
        "jit": True,
        "unroll": True,
    },
)
