from typing import cast

import jax.numpy as jnp
from jaxtyping import Array, PyTree

from evermore.parameters.parameter import Parameter, _params_map
from evermore.pdf import PDFLike

__all__ = [
    "get_log_probs",
]


def __dir__():
    return __all__


def get_log_probs(params: PyTree) -> PyTree:
    """
    Compute the log probabilities for all parameters.

    This function iterates over all parameters in the provided PyTree params,
    applies their associated prior distributions (if any), and computes the
    log probability for each parameter. If a parameter does not have a prior
    distribution, a default log probability of 0.0 is returned.

    Args:
        params (PyTree): A PyTree containing parameters to compute log probabilities for.

    Returns:
        PyTree: A PyTree with the same structure as the input, where each parameter
        is replaced by its corresponding log probability.
    """

    def _constraint(param: Parameter) -> Array:
        prior = param.prior
        if isinstance(prior, PDFLike):
            prior = cast(PDFLike, prior)
            x = prior.scale_std(param.value)
            return prior.log_prob(x)
        return jnp.array([0.0])

    # constraints from pdfs
    return _params_map(_constraint, params)
