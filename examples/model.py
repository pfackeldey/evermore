from __future__ import annotations

import jax
import jax.numpy as jnp

from dilax.model import Model, Result
from dilax.optimizer import JaxOptimizer
from dilax.parameter import Parameter, compose, lnN, modifier, shape, unconstrained
from dilax.util import HistDB


class SPlusBModel(Model):
    def __call__(
        self,
        processes: HistDB,
        parameters: dict[str, jax.Array],
    ) -> Result:
        res = Result()

        mu_modifier = modifier(
            name="mu", parameter=parameters["mu"], effect=unconstrained()
        )
        res.add(
            process="signal",
            expectation=mu_modifier(processes["signal", "nominal"]),
        )

        bkg1_modifier = compose(
            modifier(name="lnN1", parameter=parameters["norm1"], effect=lnN(0.1)),
            modifier(
                name="shape1_bkg1",
                parameter=parameters["shape1"],
                effect=shape(
                    up=processes["background1", "shape_up"],
                    down=processes["background1", "shape_down"],
                ),
            ),
        )
        res.add(
            process="background1",
            expectation=bkg1_modifier(processes["background1", "nominal"]),
        )

        bkg2_modifier = compose(
            modifier(name="lnN2", parameter=parameters["norm2"], effect=lnN(0.05)),
            modifier(
                name="shape1_bkg2",
                parameter=parameters["shape1"],
                effect=shape(
                    up=processes["background2", "shape_up"],
                    down=processes["background2", "shape_down"],
                ),
            ),
        )
        res.add(
            process="background2",
            expectation=bkg2_modifier(processes["background2", "nominal"]),
        )
        return res


def create_model():
    processes = HistDB(
        {
            ("signal", "nominal"): jnp.array([3]),
            ("background1", "nominal"): jnp.array([10]),
            ("background2", "nominal"): jnp.array([20]),
            ("background1", "shape_up"): jnp.array([12]),
            ("background1", "shape_down"): jnp.array([8]),
            ("background2", "shape_up"): jnp.array([23]),
            ("background2", "shape_down"): jnp.array([19]),
        }
    )
    parameters = {
        "mu": Parameter(value=jnp.array([1.0]), bounds=(0.0, jnp.inf)),
        "norm1": Parameter(value=jnp.array([0.0])),
        "norm2": Parameter(value=jnp.array([0.0])),
        "shape1": Parameter(value=jnp.array([0.0])),
    }

    # return model
    return SPlusBModel(processes=processes, parameters=parameters)


model = create_model()

init_values = model.parameter_values
observation = jnp.array([37])
asimov = model.evaluate().expectation()


# create optimizer (from `jaxopt`)
optimizer = JaxOptimizer.make(
    name="LBFGS",
    settings={"maxiter": 5, "jit": True, "unroll": True},
)
