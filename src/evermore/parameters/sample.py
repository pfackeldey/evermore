from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from evermore.parameters.filter import is_parameter
from evermore.parameters.parameter import (
    AbstractParameter,
    V,
    replace_value,
)
from evermore.parameters.tree import PT, only, pure
from evermore.pdf import AbstractPDF, PoissonBase
from evermore.util import _missing

__all__ = [
    "sample_from_covariance_matrix",
    "sample_from_priors",
]


def __dir__():
    return __all__


def sample_from_covariance_matrix(
    key: jax.random.PRNGKey,
    params: PT,
    *,
    covariance_matrix: Float[Array, "nparams nparams"],
    n_samples: int = 1,
) -> PT:
    """
    Samples parameter sets from a multivariate normal distribution defined by the given covariance matrix,
    centered around the current parameter values.

    See ``examples/toy_generation.py`` for an example usage.

    Args:
        key (jax.random.PRNGKey): A JAX random key used for generating random samples.
        params (PT): A PyTree of parameters whose values will be used as the mean of the distribution.
        covariance_matrix (Float[Array, "nparams nparams"]): The covariance matrix for the multivariate normal distribution.
        n_samples (int, optional): The number of samples to draw. Defaults to 1.

    Returns:
        PT: A PyTree with the same structure as `params`, where each parameter value is replaced
        by a sampled value. If `n_samples > 1`, the parameter values will have a batch dimension as the first axis.

    Example:

    .. code-block:: python

        import evermore as evm
        import jax
        import jax.numpy as jnp

        param1 = evm.Parameter(value=jnp.array([1.0]), prior=None, lower=0.0, upper=2.0)
        param2 = evm.Parameter(value=jnp.array([2.0]), prior=None, lower=1.0, upper=3.0)
        params = {"a": param1, "b": param2}
        cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        key = jax.random.PRNGKey(0)
        sampled = evm.sample.sample_from_covariance_matrix(
            key, params, covariance_matrix=cov, n_samples=3
        )
        sampled["a"].value.shape
        # (3, 1)
        sampled["b"].value.shape
        # (3, 1)
    """
    # get the value & make sure it has at least 1d so we insert a batch dim later
    params_ = only(params, is_parameter)
    values = jax.tree.map(jnp.atleast_1d, pure(params_))
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
        replace_value, params, sampled_param_values, is_leaf=is_parameter
    )


def sample_from_priors(params: PT, key: PRNGKeyArray) -> PT:
    """
    Samples from the individual prior distributions of the parameters in the given PyTree.
    Note that no correlations between parameters are taken into account during sampling.

    See ``examples/toy_generation.py`` for an example usage.

    Args:
        params (PT): A PyTree of parameters from which to sample.
        key (PRNGKeyArray): A JAX random key used for generating random samples.

    Returns:
        PT: A new PyTree with the parameters sampled from their respective prior distributions.

    Example:

    .. code-block:: python

        import evermore as evm
        import jax
        import jax.numpy as jnp


        param1 = evm.Parameter(value=jnp.array([0.0]))
        param2 = evm.NormalParameter(value=jnp.array([0.0]))
        params = {"a": param1, "b": param2}
        key = jax.random.PRNGKey(0)
        sampled = evm.sample.sample_from_priors(params, key)
        sampled["a"].value.shape
        # ()
        sampled["b"].value.shape
        # ()
    """
    flat_params, treedef = jax.tree.flatten(params, is_leaf=is_parameter)
    n_params = len(flat_params)

    # create a key for each parameter
    keys = jax.random.split(key, n_params)
    keys_tree = jax.tree.unflatten(treedef, keys)

    def _sample_from_prior(param: AbstractParameter[V], key) -> V:
        if isinstance(param.prior, AbstractPDF) and param.value is not _missing:
            pdf = param.prior

            # Sample new value from the prior pdf
            sampled_value = pdf.sample(key, shape=(1,))

            # TODO: this is not correct I assume
            if isinstance(pdf, PoissonBase):
                sampled_value = (sampled_value / pdf.lamb) - 1

            # replace in param:
            return replace_value(param, sampled_value)
        # can't sample if there's no pdf to sample from,
        # or when the value is `_missing`
        return param

    # Sample for each parameter
    return jax.tree.map(_sample_from_prior, params, keys_tree, is_leaf=is_parameter)
