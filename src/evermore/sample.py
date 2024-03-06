from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray, PyTree

from evermore.util import is_parameter


# get the PDFs from the parameters of the model
def toy_module(module: eqx.Module, key: PRNGKeyArray) -> PyTree[Callable]:
    from evermore import Parameter

    params_tree = eqx.filter(module, is_parameter, is_leaf=is_parameter)
    params_structure = jax.tree_util.tree_structure(params_tree)
    n_params = params_structure.num_leaves  # type: ignore[attr-defined]

    keys = jax.random.split(key, n_params)
    keys_tree = jax.tree_util.tree_unflatten(params_structure, keys)

    def _sample(param: Parameter, key: Parameter) -> Array:
        if not param.constraints:
            msg = f"Parameter {param} has no constraint pdf, can't sample from it."
            raise RuntimeError(msg)
        if len(param.constraints) > 1:
            msg = f"More than one constraint per parameter is not allowed. Got: {param.constraints}"
            raise ValueError(msg)
        pdf = next(iter(param.constraints))

        # sample new value from the constraint pdf
        sampled_param_value = pdf.sample(key.value, param)

        # replace the sampled parameter value and return new parameter
        return eqx.tree_at(lambda p: p.value, param, sampled_param_value)

    return jax.tree_util.tree_map(_sample, params_tree, keys_tree, is_leaf=is_parameter)
