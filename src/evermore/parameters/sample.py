from __future__ import annotations

import typing as tp

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from evermore.parameters.parameter import (
    Parameter,
    _params_map,
    _ParamsTree,
    is_parameter,
)
from evermore.pdf import PDF, PoissonBase
from evermore.util import _missing

__all__ = [
    "compute_covariance_matrix",
    "sample_from_covariance_matrix",
    "sample_from_priors",
]


def __dir__():
    return __all__


def _replace_param_value(
    param: Parameter,
    value: Float[Array, ...],
) -> Parameter:
    if param.value is _missing:
        return param
    return eqx.tree_at(lambda p: p.value, param, value)


def compute_covariance_matrix(
    loss: tp.Callable,
    params: _ParamsTree,
    *,
    args: tuple[tp.Any, ...] = (),
) -> Float[Array, "nparams nparams"]:
    r"""
    Computes the covariance matrix of the parameters under the Laplace approximation,
    by inverting the Hessian of the loss function at the current parameter values.

    See ``examples/toy_generation.py`` for an example usage.

    Args:
        loss (Callable): The loss function. Should accept (params, \*args) as arguments.
        params (_ParamsTree): A PyTree of parameters.
        args (tuple, optional): Additional arguments to pass to the loss function.

    Returns:
        Float[Array, "nparams nparams"]: The covariance matrix of the parameters.

    Example:
        >>> import evermore as evm
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>>
        >>> def loss_fn(params):
        ...     x = params["a"].value
        ...     y = params["b"].value
        ...     return jnp.sum((x - 1.0) ** 2 + (y - 2.0) ** 2)
        >>> params = {
        ...     "a": evm.Parameter(value=jnp.array([1.0]), prior=None, lower=0.0, upper=2.0),
        ...     "b": evm.Parameter(value=jnp.array([2.0]), prior=None, lower=1.0, upper=3.0),
        ... }
        >>> cov = evm.sample.compute_covariance_matrix(loss_fn, params)
        >>> cov.shape
        (2, 2)
    """
    # first, compute the hessian at the current point
    values = _params_map(lambda p: p.value, params)
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)

    def _flat_loss(flat_values: Float[Array, ...]) -> Float[Array, ""]:
        param_values = unravel_fn(flat_values)

        _params = jax.tree.map(
            _replace_param_value, params, param_values, is_leaf=is_parameter
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
    """
    Samples parameter sets from a multivariate normal distribution defined by the given covariance matrix,
    centered around the current parameter values.

    See ``examples/toy_generation.py`` for an example usage.

    Args:
        key (jax.random.PRNGKey): A JAX random key used for generating random samples.
        params (_ParamsTree): A PyTree of parameters whose values will be used as the mean of the distribution.
        covariance_matrix (Float[Array, "nparams nparams"]): The covariance matrix for the multivariate normal distribution.
        n_samples (int, optional): The number of samples to draw. Defaults to 1.

    Returns:
        _ParamsTree: A PyTree with the same structure as `params`, where each parameter value is replaced
        by a sampled value. If `n_samples > 1`, the parameter values will have a batch dimension as the first axis.

    Example:
        >>> import evermore as evm
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>>
        >>> param1 = evm.Parameter(value=jnp.array([1.0]), prior=None, lower=0.0, upper=2.0)
        >>> param2 = evm.Parameter(value=jnp.array([2.0]), prior=None, lower=1.0, upper=3.0)
        >>> params = {"a": param1, "b": param2}
        >>> cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        >>> key = jax.random.PRNGKey(0)
        >>> sampled = evm.sample.sample_from_covariance_matrix(key, params, covariance_matrix=cov, n_samples=3)
        >>> sampled["a"].value.shape
        (3, 1)
        >>> sampled["b"].value.shape
        (3, 1)
    """
    # get the value & make sure it has at least 1d so we insert a batch dim later
    values = _params_map(
        lambda p: _missing if p.value is _missing else jnp.atleast_1d(p.value), params
    )
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)

    # sample parameter sets from the correlation matrix (centered around `flat_values`)
    flat_sampled_values = jax.random.multivariate_normal(
        key=key,
        mean=flat_values,
        cov=covariance_matrix,
        shape=(n_samples,),
    )

    # insert batch dim
    sampled_param_values = jax.vmap(unravel_fn)(flat_sampled_values)

    # put them into the original structure again
    return jax.tree.map(
        _replace_param_value, params, sampled_param_values, is_leaf=is_parameter
    )


def sample_from_priors(params: _ParamsTree, key: PRNGKeyArray) -> _ParamsTree:
    """
    Samples from the individual prior distributions of the parameters in the given PyTree.
    Note that no correlations between parameters are taken into account during sampling.

    See ``examples/toy_generation.py`` for an example usage.

    Args:
        params (_ParamsTree): A PyTree of parameters from which to sample.
        key (PRNGKeyArray): A JAX random key used for generating random samples.

    Returns:
        _ParamsTree: A new PyTree with the parameters sampled from their respective prior distributions.

    Example:
        >>> import evermore as evm
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>>
        >>> param1 = evm.Parameter(value=jnp.array([0.0]))
        >>> param2 = evm.NormalParameter(value=jnp.array([0.0]))
        >>> params = {"a": param1, "b": param2}
        >>> key = jax.random.PRNGKey(0)
        >>> sampled = evm.sample.sample_from_priors(params, key)
        >>> sampled["a"].value.shape
        (1,)
        >>> sampled["b"].value.shape
        (1,)
    """
    flat_params, treedef = jax.tree.flatten(params, is_leaf=is_parameter)
    n_params = len(flat_params)

    # create a key for each parameter
    keys = jax.random.split(key, n_params)
    keys_tree = jax.tree.unflatten(treedef, keys)

    def _sample_from_prior(param: Parameter, key) -> Array:
        if isinstance(param.prior, PDF) and param.value is not _missing:
            pdf = param.prior

            # Sample new value from the prior pdf
            sampled_value = pdf.sample(key)

            # TODO: this is not correct I assume
            if isinstance(pdf, PoissonBase):
                sampled_value = (sampled_value / pdf.lamb) - 1

            # replace in param:
            return eqx.tree_at(lambda p: p.value, param, sampled_value)
        # can't sample if there's no pdf to sample from,
        # or when the value is `_missing`
        return param

    # Sample for each parameter
    return jax.tree.map(_sample_from_prior, params, keys_tree, is_leaf=is_parameter)
