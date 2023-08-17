import jax
import jax.numpy as jnp
import equinox as eqx

from functools import partial

from dilax.likelihood import NLL
from dilax.model import Model
from dilax.optimizer import JaxOptimizer


from examples.model import model, observation, optimizer

from jax.config import config

config.update("jax_enable_x64", True)


def nll_profiling(
    value_name: str,
    scan_points: jax.Array,
    model: Model,
    observation: jax.Array,
    optimizer: JaxOptimizer,
) -> jax.Array:
    # define single fit for a fixed parameter of interest (poi)
    @partial(jax.jit, static_argnames=("value_name", "optimizer"))
    def fixed_poi_fit(
        value_name: str,
        scan_point: jax.Array,
        model: Model,
        observation: jax.Array,
        optimizer: JaxOptimizer,
    ) -> jax.Array:
        # fix theta into the model
        model = model.update(values={value_name: scan_point})
        init_values = model.parameter_values
        init_values.pop(value_name, 1)
        # minimize
        nll = eqx.filter_jit(NLL(model=model, observation=observation))
        values, _ = optimizer.fit(fun=nll, init_values=init_values)
        return nll(values=values)

    # vectorise for multiple fixed values (scan points)
    fixed_poi_fit_vec = jax.vmap(fixed_poi_fit, in_axes=(None, 0, None, None, None))
    return fixed_poi_fit_vec(value_name, scan_points, model, observation, optimizer)


# profile the NLL around starting point of `0`
profile = nll_profiling(
    value_name="norm2",
    scan_points=jnp.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]),
    model=model,
    observation=observation,
    optimizer=optimizer,
)
