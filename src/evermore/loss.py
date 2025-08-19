import typing as tp

import jax
import jax.flatten_util
import jax.numpy as jnp
from jaxtyping import Array, Float

from evermore.parameters.filter import is_parameter
from evermore.parameters.parameter import (
    AbstractParameter,
    V,
    replace_value,
)
from evermore.parameters.tree import PT, combine, only, partition, pure
from evermore.pdf import AbstractPDF, ImplementsFromUnitNormalConversion, Normal

__all__ = [
    "covariance_matrix",
    "cramer_rao_uncertainty",
    "fisher_information_matrix",
    "get_log_probs",
    "hessian_matrix",
]


def __dir__():
    return __all__


def get_log_probs(tree: PT) -> PT:
    """
    Compute the log probabilities for all parameters.

    This function iterates over all parameters in the provided PyTree tree,
    applies their associated prior distributions (if any), and computes the
    log probability for each parameter. If a parameter does not have a prior
    distribution, a default log probability of 0.0 is returned.

    Args:
        tree (PyTree): A PyTree containing parameters to compute log probabilities for.

    Returns:
        PyTree: A PyTree with the same structure as the input, where each parameter
        is replaced by its corresponding log probability.
    """

    def _constraint(param: AbstractParameter[V]) -> V:
        prior: AbstractPDF | None = param.prior

        # unconstrained case is easy:
        if prior is None:
            return jnp.zeros_like(param.value)

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
            # this is a general implementation to translate from a unit normal to any target AbstractPDF
            # the only requirement is that the target pdf implements `.inv_cdf`.
            unit_normal: Normal[V] = Normal(
                mean=jnp.zeros_like(param.value), width=jnp.ones_like(param.value)
            )
            cdf = unit_normal.cdf(param.value)
            x = prior.inv_cdf(cdf)
        return prior.log_prob(x)

    # constraints from pdfs
    return jax.tree.map(_constraint, only(tree, is_parameter), is_leaf=is_parameter)


def _ravel_pure_tree(tree: PT) -> tuple[Float[Array, " nparams"], tp.Callable]:
    """Flattens a PyTree of parameters into a 1D array of parameter values.
    Args:
        tree (PT): A PyTree of parameters.

    Returns:
        tuple[Float[Array, "nparams"], Callable]: A tuple containing the flattened
        array of parameter values and a function to unflatten the array back into
        the original PyTree structure.
    """
    values = pure(tree)
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)
    return flat_values, unravel_fn


def hessian_matrix(
    loss_fn: tp.Callable,
    tree: PT,
) -> Float[Array, "nparams nparams"]:
    """
    Computes the Hessian matrix of the loss function at the current parameter values.

    Args:
        loss_fn (Callable): The loss function. Should accept (tree) as arguments.
            All other arguments have to be "partial'd" into the loss function.
        tree (PT): A PyTree of parameters.

    Returns:
        Float[Array, "nparams nparams"]: The Hessian matrix of the loss function.

    Example:

    .. code-block:: python

        import jax.numpy as jnp


        def loss_fn(params):
            x = params["a"].value
            y = params["b"].value
            return jnp.sum((x - 1.0) ** 2 + (y - 2.0) ** 2)


        params = {
            "a": evm.Parameter(value=jnp.array([1.0])),
            "b": evm.Parameter(value=jnp.array([2.0])),
        }

        hessian = evm.loss.hessian(loss_fn, params)
        hessian.shape
        # (2, 2)
    """
    flat_values, unravel_fn = _ravel_pure_tree(tree)

    def _flat_loss(flat_values: Float[Array, "..."]) -> Float[Array, ""]:
        param_values = unravel_fn(flat_values)

        # update the parameters with the new values
        # and call the loss function
        # 1. partition the tree of parameters and other things
        # 2. update the parameters with the new values
        # 3. combine the updated parameters with the rest of the tree
        # 4. call the loss function with the updated tree
        params, rest = partition(tree, filter=is_parameter)

        updated_params = jax.tree.map(
            replace_value, params, param_values, is_leaf=is_parameter
        )

        updated_tree = combine(updated_params, rest)
        return loss_fn(updated_tree)

    # calculate hessian
    return jax.hessian(_flat_loss)(flat_values)


def fisher_information_matrix(
    loss_fn: tp.Callable,
    tree: PT,
) -> Float[Array, "nparams nparams"]:
    """
    Computes the Fisher information matrix of the parameters under the Laplace approximation,
    by computing the Hessian of the loss function at the current parameter values.

    Args:
        loss_fn (Callable): The loss function. Should accept (tree) as arguments.
            All other arguments have to be "partial'd" into the loss function.
        tree (PT): A PyTree of parameters.

    Returns:
        Float[Array, "nparams nparams"]: The Fisher information matrix of the parameters.

    Example:

    .. code-block:: python

        import jax.numpy as jnp


        def loss_fn(params):
            x = params["a"].value
            y = params["b"].value
            return jnp.sum((x - 1.0) ** 2 + (y - 2.0) ** 2)


        params = {
            "a": evm.Parameter(value=jnp.array([1.0])),
            "b": evm.Parameter(value=jnp.array([2.0])),
        }

        fisher = evm.loss.fisher_information_matrix(loss_fn, params)
        fisher.shape
        # (2, 2)
    """
    # calculate hessian
    hessian = hessian_matrix(loss_fn, tree)
    # invert to get the fisher information matrix under the Laplace assumption of normality
    return jnp.linalg.inv(hessian)


def covariance_matrix(
    loss_fn: tp.Callable,
    tree: PT,
) -> Float[Array, "nparams nparams"]:
    """
    Computes the covariance matrix of the parameters under the Laplace approximation,
    by inverting the Hessian of the loss function at the current parameter values.

    See ``examples/toy_generation.py`` for an example usage.

    Args:
        loss_fn (Callable): The loss function. Should accept (tree) as arguments.
            All other arguments have to be "partial'd" into the loss function.
        tree (PT): A PyTree of parameters.

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
            "a": evm.Parameter(value=jnp.array([1.0])),
            "b": evm.Parameter(value=jnp.array([2.0])),
        }

        cov = evm.loss.covariance(loss_fn, params)
        cov.shape
        # (2, 2)
    """
    # calculate fisher information matrix
    fisher = fisher_information_matrix(loss_fn, tree)

    # normalize via D^-1 @ fisher @ D^-1 with D being the diagnonal standard deviation matrix
    d = jnp.sqrt(jnp.diag(fisher))
    cov = fisher / jnp.outer(d, d)

    # to avoid numerical issues, fix the diagonal to 1
    return jnp.fill_diagonal(cov, 1.0, inplace=False)


def cramer_rao_uncertainty(
    loss_fn: tp.Callable,
    tree: PT,
) -> PT:
    """
    Computes the Cramer-Rao uncertainty of the parameters under the Laplace approximation,
    by computing the square root of the diagonal of the Fisher information matrix.

    Args:
        loss_fn (Callable): The loss function. Should accept (tree) as arguments.
            All other arguments have to be "partial'd" into the loss function.
        tree (PT): A PyTree of parameters.

    Returns:
        PT: The Cramer-Rao uncertainty of the parameters.

    Example:

    .. code-block:: python

        import jax.numpy as jnp


        def loss_fn(params):
            x = params["a"].value
            y = params["b"].value
            return jnp.sum((x - 1.0) ** 2 + (y - 2.0) ** 2)


        params = {
            "a": evm.Parameter(value=jnp.array([1.0])),
            "b": evm.Parameter(value=jnp.array([2.0])),
        }

        uncertainties = evm.loss.cramer_rao_uncertainty(loss_fn, params)
    """
    _, unravel_fn = _ravel_pure_tree(tree)

    # calculate fisher information matrix
    fisher_info = fisher_information_matrix(loss_fn, tree)
    return unravel_fn(jnp.sqrt(jnp.diag(fisher_info)))
