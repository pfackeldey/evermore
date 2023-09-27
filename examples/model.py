from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from dilax.model import Model, Result
from dilax.optimizer import JaxOptimizer
from dilax.parameter import Parameter, lnN, modifier, unconstrained
from dilax.util import HistDB


class SPlusBModel(Model):
    def __call__(
        self,
        processes: HistDB,
        parameters: dict[str, jax.Array],
    ) -> Result:
        res = Result()

        res.add(
            process="signal",
            expectation=modifier(
                name="mu",
                parameter=parameters["mu"],
                effect=unconstrained(),
            )(processes["signal"]),
        )
        res.add(
            process="background",
            expectation=modifier(
                name="lnN1",
                parameter=parameters["norm1"],
                effect=lnN(0.1),
            )(processes["background1"]),
        )
        res.add(
            process="background2",
            expectation=modifier(
                name="lnN2",
                parameter=parameters["norm1"],
                effect=lnN(0.05),
            )(processes["background2"]),
        )
        return res


def create_model():
    processes = HistDB(
        {
            "signal": jnp.array([3]),
            "background1": jnp.array([10]),
            "background2": jnp.array([20]),
        }
    )
    parameters = {
        "mu": Parameter(value=jnp.array([1.0]), bounds=(0.0, jnp.inf)),
        "norm1": Parameter(value=jnp.array([0.0]), bounds=(-jnp.inf, jnp.inf)),
        "norm2": Parameter(value=jnp.array([0.0]), bounds=(-jnp.inf, jnp.inf)),
    }

    # return model
    return SPlusBModel(processes=processes, parameters=parameters)


model = create_model()

eqx.tree_pprint(model)

init_values = model.parameter_values
observation = jnp.array([37])


# create optimizer (from `jaxopt`)
optimizer = JaxOptimizer.make(
    name="LBFGS",
    settings={"maxiter": 5, "jit": True, "unroll": True},
)
