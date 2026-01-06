from __future__ import annotations

import jax
import jax.flatten_util
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, PyTree

from evermore.parameters.filter import is_dynamic_parameter, is_parameter
from evermore.parameters.parameter import BaseParameter, V
from evermore.pdf import BasePDF, PoissonBase

__all__ = [
    "sample_from_covariance_matrix",
    "sample_from_priors",
]


def __dir__():
    return __all__


def sample_from_covariance_matrix(
    rngs: nnx.Rngs,
    params: PyTree[BaseParameter],
    *,
    covariance_matrix: Float[Array, "nparams nparams"],
    mask: PyTree[bool] | None = None,
    n_samples: int = 1,
) -> PyTree[BaseParameter]:
    """Samples new parameter configurations from a multivariate normal.

    Args:
        rngs: ``nnx.Rngs`` container used to draw randomness.
        params: PyTree of parameters providing the mean values.
        covariance_matrix: Covariance matrix defining the multivariate normal.
        mask: Optional PyTree indicating which parameters should be resampled.
        n_samples: Number of samples to draw; adds a leading batch dimension when ``> 1``.

    Returns:
        PyTree with sampled parameter values replacing the originals.

    Examples:
        >>> import evermore as evm
        >>> import jax.numpy as jnp
        >>> from flax import nnx
        >>> cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        >>> params = {
        ...     "a": evm.Parameter(value=jnp.array([1.0])),
        ...     "b": evm.Parameter(value=jnp.array([2.0])),
        ... }
        >>> rngs = nnx.Rngs(0)
        >>> samples = evm.sample.sample_from_covariance_matrix(
        ...     rngs, params, covariance_matrix=cov, n_samples=3
        ... )
        >>> samples["a"].value.shape
        (3, 1)
    """
    # get the value & make sure it has at least 1d so we insert a batch dim later
    graphdef, params_state, rest = nnx.split(params, is_dynamic_parameter, ...)
    values = jax.tree.map(jnp.atleast_1d, nnx.pure(params_state))
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)

    # sample parameter sets from the correlation matrix (centered around `flat_values`)
    flat_sampled_values = rngs.multivariate_normal(
        mean=flat_values,
        cov=covariance_matrix,
        shape=(n_samples,),
    )

    # insert batch dim
    sampled_param_values = jax.vmap(unravel_fn)(flat_sampled_values)

    def _update(path, variable, value):
        del path  # unused
        return variable.replace(value=value)

    # using jax.tree.map here to not do inplace updates
    sampled_params_state = jax.tree.map_with_path(
        _update,
        params_state,
        sampled_param_values,
        is_leaf=is_parameter,
        is_leaf_takes_path=True,
    )
    return nnx.merge(graphdef, sampled_params_state, rest)


def sample_from_priors(
    rngs: nnx.Rngs, params: PyTree[BaseParameter]
) -> PyTree[BaseParameter]:
    """Samples independent values from each parameter's prior distribution.

    Args:
        rngs: ``nnx.Rngs`` container used to draw randomness.
        params: PyTree containing the parameters to sample.

    Returns:
        PyTree mirroring ``params`` with sampled values substituted in place of ``.value``.

    Examples:
        >>> import evermore as evm
        >>> import jax
        >>> from flax import nnx
        >>> params = {
        ...     "a": evm.Parameter(value=0.0),
        ...     "b": evm.NormalParameter(value=0.0),
        ... }
        >>> samples = evm.sample.sample_from_priors(nnx.Rngs(0), params)
        >>> isinstance(samples["b"].value, jax.Array)
        True
    """
    graphdef, params_state, rest = nnx.split(params, is_parameter, ...)

    def _sample_from_prior(path, param: BaseParameter[V]) -> BaseParameter[V]:
        del path  # unused
        if isinstance(param.prior, BasePDF):
            pdf = param.prior

            # Sample new value from the prior pdf
            sampled_value = pdf.sample(rngs(), shape=param.value.shape)

            # TODO: this is not correct I assume
            if isinstance(pdf, PoissonBase):
                sampled_value = (sampled_value / pdf.lamb) - 1

            return param.replace(value=sampled_value)  # ty:ignore[invalid-return-type]
        # can't sample if there's:
        # - no pdf to sample from
        return param

    # Sample for each parameter
    sampled_params_state = nnx.map_state(_sample_from_prior, params_state)
    return nnx.merge(graphdef, sampled_params_state, rest)
