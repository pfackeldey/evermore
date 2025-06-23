import typing as tp

import jax
import jax.flatten_util
import jax.numpy as jnp
from jaxtyping import Array, Float

from evermore.parameters.parameter import (
    Parameter,
    _params_map,
    _ParamsTree,
    is_parameter,
    replace_value,
)
from evermore.pdf import PDF, ImplementsFromUnitNormalConversion, Normal

__all__ = [
    "compute_covariance",
    "get_log_probs",
]


def __dir__():
    return __all__


def get_log_probs(params: _ParamsTree) -> _ParamsTree:
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

    def _constraint(param: Parameter) -> Float[Array, "..."]:
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


def compute_covariance(
    loss_fn: tp.Callable,
    params: _ParamsTree,
) -> Float[Array, "nparams nparams"]:
    r"""
    Computes the covariance matrix of the parameters under the Laplace approximation,
    by inverting the Hessian of the loss function at the current parameter values.

    See ``examples/toy_generation.py`` for an example usage.

    Args:
        loss_fn (Callable): The loss function. Should accept (params) as arguments.
            All other arguments have to be "partial'd" into the loss function.
        params (_ParamsTree): A PyTree of parameters.

    Returns:
        Float[Array, "nparams nparams"]: The covariance matrix of the parameters.

    Example:

    .. code-block:: python

        import evermore as evm
        import jax
        import jax.numpy as jnp


        def loss_fn(params):
            x = params["a"].value
            y = params["b"].value
            return jnp.sum((x - 1.0) ** 2 + (y - 2.0) ** 2)


        params = {
            "a": evm.Parameter(value=jnp.array([1.0]), prior=None, lower=0.0, upper=2.0),
            "b": evm.Parameter(value=jnp.array([2.0]), prior=None, lower=1.0, upper=3.0),
        }

        cov = evm.loss.compute_covariance(loss_fn, params)
        cov.shape
        # (2, 2)
    """
    # first, compute the hessian at the current point
    values = _params_map(lambda p: p.value, params)
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)

    def _flat_loss(flat_values: Float[Array, "..."]) -> Float[Array, ""]:
        param_values = unravel_fn(flat_values)

        _params = jax.tree.map(
            replace_value, params, param_values, is_leaf=is_parameter
        )
        return loss_fn(_params)

    # calculate hessian
    hessian = jax.hessian(_flat_loss)(flat_values)

    # invert to get the correlation matrix under the Laplace assumption of normality
    cov = jnp.linalg.inv(hessian)

    # normalize via D^-1 @ cov @ D^-1 with D being the diagnonal standard deviation matrix
    d = jnp.sqrt(jnp.diag(cov))
    cov = cov / jnp.outer(d, d)

    # to avoid numerical issues, fix the diagonal to 1
    return jnp.fill_diagonal(cov, 1.0, inplace=False)
