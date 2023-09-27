from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.config import config
from model import asimov, model, optimizer

from dilax.likelihood import NLL
from dilax.model import Model
from dilax.optimizer import JaxOptimizer

config.update("jax_enable_x64", True)


def nll_profiling(
    value_name: str,
    scan_points: jax.Array,
    model: Model,
    observation: jax.Array,
    optimizer: JaxOptimizer,
    fit: bool,
) -> jax.Array:
    # define single fit for a fixed parameter of interest (poi)
    @partial(jax.jit, static_argnames=("value_name", "optimizer", "fit"))
    def fixed_poi_fit(
        value_name: str,
        scan_point: jax.Array,
        model: Model,
        observation: jax.Array,
        optimizer: JaxOptimizer,
        fit: bool,
    ) -> jax.Array:
        # fix theta into the model
        model = model.update(values={value_name: scan_point})
        init_values = model.parameter_values
        init_values.pop(value_name, 1)
        # minimize
        nll = eqx.filter_jit(NLL(model=model, observation=observation))
        if fit:
            values, _ = optimizer.fit(fun=nll, init_values=init_values)
        else:
            values = model.parameter_values
        return nll(values=values)

    # vectorise for multiple fixed values (scan points)
    fixed_poi_fit_vec = jax.vmap(
        fixed_poi_fit, in_axes=(None, 0, None, None, None, None)
    )
    return fixed_poi_fit_vec(
        value_name, scan_points, model, observation, optimizer, fit
    )


# profile the NLL around starting point of `0`
scan_points = jnp.r_[-1.9:2.0:0.1]

profile_postfit = nll_profiling(
    value_name="norm1",
    scan_points=scan_points,
    model=model,
    observation=asimov,
    optimizer=optimizer,
    fit=True,
)
