from __future__ import annotations

import typing as tp

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from evermore.parameters.parameter import (
    _params_map,
    _ParamsTree,
    is_parameter,
)

__all__ = [
    "compute_covariance_matrix",
    "sample_from_covariance_matrix",
]


def __dir__():
    return __all__


def compute_covariance_matrix(
    loss: tp.Callable,
    params: _ParamsTree,
    *,
    args: tuple[tp.Any, ...] = (),
) -> Float[Array, "nparams nparams"]:
    # first, compute the hessian at the current point
    values = _params_map(lambda p: p.value, params)
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)

    def _flat_loss(flat_values: Float[Array, ...]) -> Float[Array, ""]:
        param_values = unravel_fn(flat_values)

        def _replace_value(param, value):
            return eqx.tree_at(lambda p: p.value, param, value)

        _params = jax.tree.map(
            _replace_value, params, param_values, is_leaf=is_parameter
        )
        return loss(_params, *args)

    # calculate hessian
    hessian = jax.hessian(_flat_loss)(flat_values)

    # invert to get the correlation matrix under the Laplace assumption of normality
    return jnp.linalg.inv(hessian)


def sample_from_covariance_matrix(
    key: jax.random.PRNGKey,
    params: _ParamsTree,
    *,
    covariance_matrix: Float[Array, "nparams nparams"],
    n_samples: int = 1,
) -> _ParamsTree:
    # get the value & make sure it has at least 1d so we insert a batch dim later
    values = _params_map(lambda p: jnp.atleast_1d(p.value), params)
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)

    # sample parameter sets from the correlation matrix
    # note that the sampling should be centered around the current parameters,
    # but 0 is chosen here since they are used as offsets
    flat_sampled_offsets = jax.random.multivariate_normal(
        key=key,
        mean=jnp.zeros_like(flat_values),
        cov=covariance_matrix,
        shape=(n_samples,),
    )

    # insert batch dim
    sampled_param_values = jax.vmap(unravel_fn, in_axes=(0,))(
        flat_values + flat_sampled_offsets
    )

    # put them into the original structure again
    def _replace_value(param, value):
        return eqx.tree_at(lambda p: p.value, param, value)

    return jax.tree.map(
        _replace_value, params, sampled_param_values, is_leaf=is_parameter
    )
