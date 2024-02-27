from __future__ import annotations

import jax.numpy as jnp

import evermore as evm


class SPlusBModel(evm.Model):
    def __call__(self, processes: dict, parameters: dict) -> evm.Result:
        res = evm.Result()

        mu_modifier = evm.modifier(
            name="mu", parameter=parameters["mu"], effect=evm.effect.unconstrained()
        )
        res.add(
            process="signal",
            expectation=mu_modifier(processes[("signal", "nominal")]),
        )

        bkg1_modifier = evm.compose(
            evm.modifier(
                name="lnN1",
                parameter=parameters["norm1"],
                effect=evm.effect.lnN((0.9, 1.1)),
            ),
            evm.modifier(
                name="shape1_bkg1",
                parameter=parameters["shape1"],
                effect=evm.effect.shape(
                    up=processes[("background1", "shape_up")],
                    down=processes[("background1", "shape_down")],
                ),
            ),
        )
        res.add(
            process="background1",
            expectation=bkg1_modifier(processes[("background1", "nominal")]),
        )

        bkg2_modifier = evm.compose(
            evm.modifier(
                name="lnN2",
                parameter=parameters["norm2"],
                effect=evm.effect.lnN((0.95, 1.05)),
            ),
            evm.modifier(
                name="shape1_bkg2",
                parameter=parameters["shape1"],
                effect=evm.effect.shape(
                    up=processes[("background2", "shape_up")],
                    down=processes[("background2", "shape_down")],
                ),
            ),
        )
        res.add(
            process="background2",
            expectation=bkg2_modifier(processes[("background2", "nominal")]),
        )
        return res


def create_model():
    processes = {
        ("signal", "nominal"): jnp.array([3]),
        ("background1", "nominal"): jnp.array([10]),
        ("background2", "nominal"): jnp.array([20]),
        ("background1", "shape_up"): jnp.array([12]),
        ("background1", "shape_down"): jnp.array([8]),
        ("background2", "shape_up"): jnp.array([23]),
        ("background2", "shape_down"): jnp.array([19]),
    }
    parameters = {
        "mu": evm.Parameter(value=jnp.array([1.0]), bounds=(0.0, jnp.inf)),
        "norm1": evm.Parameter(value=jnp.array([0.0])),
        "norm2": evm.Parameter(value=jnp.array([0.0])),
        "shape1": evm.Parameter(value=jnp.array([0.0])),
    }

    # return model
    return SPlusBModel(processes=processes, parameters=parameters)


model = create_model()

init_values = model.parameter_values
observation = jnp.array([37])
asimov = model.evaluate().expectation()


# create optimizer (from `jaxopt`)
optimizer = evm.optimizer.JaxOptimizer.make(
    name="LBFGS",
    settings={"maxiter": 5, "jit": True, "unroll": True},
)
