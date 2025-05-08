from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from evermore.parameters.parameter import Parameter, _ParamsTree, is_parameter
from evermore.pdf import PDF, PoissonBase

__all__ = [
    "sample_uncorrelated",
]


def __dir__():
    return __all__


def sample_uncorrelated(params: _ParamsTree, key: PRNGKeyArray) -> _ParamsTree:
    """
    Samples from the individual prior distributions of the parameters in the given PyTree.
    Note that no correlations between parameters are taken into account during sampling.

    Args:
        params (_ParamsTree): A PyTree of parameters from which to sample.
        key (PRNGKeyArray): A JAX random key used for generating random samples.

    Returns:
        _ParamsTree: A new PyTree with the parameters sampled from their respective prior distributions.

    Example:
        See examples/toy_generation.py for an example usage.
    """
    # Partition the tree into parameters and the rest
    params_tree, rest_tree = eqx.partition(params, is_parameter, is_leaf=is_parameter)
    params_structure = jax.tree.structure(params_tree)
    n_params = params_structure.num_leaves  # type: ignore[attr-defined]

    keys = jax.random.split(key, n_params)
    keys_tree = jax.tree.unflatten(params_structure, keys)

    def _sample(param: Parameter, key: Parameter) -> Array:
        if isinstance(param.prior, PDF):
            pdf = param.prior

            # Sample new value from the prior pdf
            sampled_value = pdf.sample(key.value)

            # TODO: Make this compatible with externally provided Poisson PDFs
            if isinstance(pdf, PoissonBase):
                sampled_value = (sampled_value / pdf.lamb) - 1
        else:
            assert param.prior is None, f"Unknown prior type: {param.prior}."
            msg = f"Can't sample uniform from {param} (no given prior). "
            param = eqx.error_if(
                param, ~jnp.isfinite(param.lower), msg + "No lower bound given."
            )
            param = eqx.error_if(
                param, ~jnp.isfinite(param.upper), msg + "No upper bound given."
            )
            sampled_value = jax.random.uniform(
                key.value,
                shape=param.value.shape,
                minval=param.lower,
                maxval=param.upper,
            )

        # Replace the sampled parameter value and return new parameter
        return eqx.tree_at(lambda p: p.value, param, sampled_value)

    # Sample for each parameter
    sampled_params_tree = jax.tree.map(
        _sample, params_tree, keys_tree, is_leaf=is_parameter
    )

    # Combine the sampled parameters with the rest of the model and return it
    return eqx.combine(sampled_params_tree, rest_tree, is_leaf=is_parameter)
