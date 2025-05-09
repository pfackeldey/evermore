import collections
import typing as tp

import jax
import jax.flatten_util
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from evermore.parameters.parameter import Parameter, _params_map
from evermore.pdf import PDF, ImplementsFromUnitNormalConversion, Normal

__all__ = [
    "compute_correlation",
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
        prior: PDF | None = param.prior

        # unconstrained case is easy:
        if prior is None:
            return jnp.array([0.0])

        # all constrained parameters are 'moving' on a 'unit_normal' distribution (mean=0, width=1), i.e.:
        # - param.value=0: no shift, no constrain
        # - param.value=+1: +1 sigma shift, calculate the +1 sigma constrain based on prior pdf
        # - param.value=-1: -1 sigma shift, calculate the -1 sigma constrain based on prior pdf
        #
        # Translating between this "unit_normal" pdf and any other pdf works as follows:
        # x = AnyOtherPDF.inv_cdf(unit_normal.cdf(v))
        #
        # Some priors, such as other Normals, can do a shortcut to save compute:
        # e.g. for Normal: x = mean + width * v
        # these shortcuts are implemented by '__evermore_from_unit_normal__' as defined by the
        # ImplementsFromUnitNormalConversion protocol.
        #
        # (in the following: x=x and v=param.value)
        if isinstance(prior, ImplementsFromUnitNormalConversion):
            # this is the fast-path
            x = prior.__evermore_from_unit_normal__(param.value)
        else:
            # this is a general implementation to translate from a unit normal to any target PDF
            # the only requirement is that the target pdf implements `.inv_cdf`.
            unit_normal = Normal(
                mean=jnp.zeros_like(param.value), width=jnp.ones_like(param.value)
            )
            cdf = unit_normal.cdf(param.value)
            x = prior.inv_cdf(cdf)
        return prior.log_prob(x)

    # constraints from pdfs
    return _params_map(_constraint, params)


def compute_correlation(
    loss_fn: collections.abc.Callable,
    params: PyTree,
    args: tuple[tp.Any, ...] = (),
) -> Array:
    # create a flattened version of the parameters and the loss
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)

    def flat_loss_fn(flat_params: Array) -> float:
        return loss_fn(unravel_fn(flat_params), *args)

    # compute the hessian at the current parameters
    h = jax.hessian(flat_loss_fn)(flat_params)

    # invert to get the correlation matrix under the Laplace assumption of normality
    return jnp.linalg.inv(h)
