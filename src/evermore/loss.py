import jax.numpy as jnp
from jaxtyping import Array, PyTree

from evermore.parameters.parameter import Parameter, _params_map
from evermore.pdf import PDF, Normal

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
        if prior is None:
            return jnp.array([0.0])

        assert isinstance(prior, PDF), f"Prior must be a PDF object, got {type(prior)}"

        # all constrained parameters are 'moving' on a 'unit_normal' distribution (mean=0, width=1), ie:
        # - param.value=0: no shift, no constrain
        # - param.value=+1: +1 sigma shift, calculate the +1 sigma constrain based on prior pdf
        # - param.value=-1: -1 sigma shift, calculate the -1 sigma constrain based on prior pdf
        # Translating between this "unit_normal" pdf and any other pdf works as follows:
        # x' = AnyOtherPDF.inv_cdf(unit_normal.cdf(x))
        unit_normal = Normal(
            mean=jnp.zeros_like(param.value), width=jnp.ones_like(param.value)
        )
        cdf = unit_normal.cdf(param.value)
        x = prior.inv_cdf(cdf)
        return prior.log_prob(x)

    # constraints from pdfs
    return _params_map(_constraint, params)
