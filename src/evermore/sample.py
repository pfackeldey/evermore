from typing import cast

import equinox as eqx
import jax
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, PyTree

from evermore.custom_types import PDFLike
from evermore.parameter import Parameter
from evermore.pdf import Poisson
from evermore.util import is_parameter

__all__ = [
    "sample_parameters",
]


def __dir__():
    return __all__


def sample_parameters(tree: PyTree, key: PRNGKeyArray) -> PyTree:
    # partition the tree into parameters and the rest
    params_tree, rest_tree = eqx.partition(tree, is_parameter, is_leaf=is_parameter)
    params_structure = jax.tree_util.tree_structure(params_tree)
    n_params = params_structure.num_leaves  # type: ignore[attr-defined]

    keys = jax.random.split(key, n_params)
    keys_tree = jax.tree_util.tree_unflatten(params_structure, keys)

    def _sample(param: Parameter, key: Parameter) -> Array:
        if not isinstance(param.constraint, PDFLike):
            msg = f"Parameter {param} has no sampling method, can't sample from it."
            raise RuntimeError(msg)

        pdf = param.constraint
        pdf = cast(PDFLike, pdf)

        # sample new value from the constraint pdf
        sampled_value = pdf.sample(key.value)

        # TODO: make this compatible with externally provided Poisson PDFs
        if isinstance(pdf, Poisson):
            sampled_value = (sampled_value / pdf.lamb) - 1

        # replace the sampled parameter value and return new parameter
        return eqx.tree_at(lambda p: p.value, param, sampled_value)

    # sample for each parameter
    sampled_params_tree = jtu.tree_map(
        _sample, params_tree, keys_tree, is_leaf=is_parameter
    )

    # combine the sampled parameters with the rest of the model and return it
    return eqx.combine(sampled_params_tree, rest_tree, is_leaf=is_parameter)
