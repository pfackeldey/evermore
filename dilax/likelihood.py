from functools import partial

import jax
import jax.numpy as jnp

from dilax.model import Model


@jax.jit
def nll(
    parameters: dict[str, jax.Array],
    model: Model,
    observation: jax.Array,
) -> jax.Array:
    """Negative log-likelihood for Poisson distribution."""

    expectation = model.apply(parameters=parameters).eval()
    constraint = model.apply(parameters=parameters).nll_constraint()

    nll = jax.scipy.stats.poisson.logpmf(observation, expectation) + constraint

    # negative log-likelihood
    return -jnp.sum(nll, axis=-1)


@jax.jit
def cov_matrix(
    parameters: dict[str, jax.Array],
    model: Model,
    observation: jax.Array,
) -> jax.Array:
    hessian = jax.hessian(nll, argnums=0)(parameters, model, observation)
    hessian, _ = jax.tree_util.tree_flatten(hessian)
    hessian = jnp.array(hessian)
    new_shape = len(parameters)
    hessian = jnp.reshape(hessian, (new_shape, new_shape))
    hessian_inv = jnp.linalg.inv(hessian)
    return hessian_inv


@partial(jax.jit, static_argnames=("toys",))
def sample(
    parameters: dict[str, jax.Array],
    model: Model,
    observation: jax.Array,
    toys: int,
) -> jax.Array:
    key = jax.random.PRNGKey(1234)

    @jax.jit
    def _gen(
        key: jax.Array,
        parameters: dict[str, jax.Array],
        model: Model,
        observation: jax.Array,
    ) -> jax.Array:
        cov = cov_matrix(parameters=parameters, model=model, observation=observation)
        parameters, tree_def = jax.tree_util.tree_flatten(parameters)
        params = jax.random.multivariate_normal(key=key, mean=jnp.array(parameters), cov=cov)
        params = jax.tree_util.tree_unflatten(tree_def, params)
        return model.apply(parameters=params).eval()

    return jax.vmap(_gen, in_axes=(0, None, None, None))(
        jax.random.split(key, num=toys), parameters, model, observation
    )
