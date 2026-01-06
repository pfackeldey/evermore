import typing as tp

import jax
import jax.flatten_util
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, PyTree

from evermore.parameters.filter import is_dynamic_parameter, is_parameter
from evermore.parameters.parameter import BaseParameter, V
from evermore.pdf import BasePDF, ImplementsFromUnitNormalConversion, Normal

__all__ = [
    "covariance_matrix",
    "cramer_rao_uncertainty",
    "fisher_information_matrix",
    "get_log_probs",
    "hessian_matrix",
]


def __dir__():
    return __all__


def _parameter_value_to_x(prior: BasePDF, value: V) -> V:
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
        x = prior.__evermore_from_unit_normal__(value)
    else:
        # this is a general implementation to translate from a unit normal to any target BasePDF
        # the only requirement is that the target pdf implements `.inv_cdf`.
        unit_normal = Normal(mean=jnp.zeros_like(value), width=jnp.ones_like(value))
        cdf = unit_normal.cdf(value)
        x = prior.inv_cdf(cdf)
    return x


def get_log_probs(tree: PyTree[BaseParameter]) -> nnx.State:
    """Computes log probabilities for every parameter in a PyTree.

    The function iterates over each parameter, evaluates its prior distribution
    (if present), and returns an ``nnx.State`` whose leaves store the
    corresponding log probabilities.

    Args:
        tree: PyTree that may contain parameters and auxiliary nodes.

    Returns:
        nnx.State: State matching the input structure with log probabilities in
            place of the original parameters.
    """
    params_state, _ = nnx.state(tree, is_parameter, ...)

    def _constraint(path, param: BaseParameter[V]) -> V:
        del path  # unused
        prior: BasePDF | None = param.prior
        # unconstrained case is easy:
        if prior is None:
            return jnp.zeros_like(param.get_value())  # ty:ignore[invalid-return-type]

        # constrained case:
        if not isinstance(param.prior, BasePDF):
            msg = f"Prior must be a BasePDF object for a constrained BaseParameter (or 'None' for an unconstrained one), got {param.prior=} ({type(param.prior)=})"  # type: ignore[unreachable]
            raise ValueError(msg)

        x = _parameter_value_to_x(prior, param.get_value())
        return prior.log_prob(x)

    # constraints from pdfs
    return nnx.map_state(_constraint, params_state)


def _ravel_pure_tree(
    tree: PyTree[BaseParameter],
) -> tuple[Float[Array, " nparams"], tp.Callable]:
    """Flattens a PyTree of parameters into a 1D array of parameter values.

    Args:
        tree: PyTree of parameters.

    Returns:
        tuple[Float[Array, "nparams"], tp.Callable]: Pair containing the flattened
            parameter values and a function that reconstructs the original PyTree.
    """
    values = nnx.pure(tree)
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)
    return flat_values, unravel_fn


def hessian_matrix(
    loss_fn: tp.Callable,
    tree: PyTree[BaseParameter],
) -> Float[Array, "nparams nparams"]:
    """Computes the Hessian of a scalar loss with respect to dynamic parameters.

    The function leverages ``flax.nnx.split`` to separate differentiable and
    static state and evaluates the Hessian of ``loss_fn`` at the current
    parameter values.

    Args:
        loss_fn: Callable that accepts a PyTree of parameters and returns a scalar loss.
        tree: PyTree containing parameters and auxiliary nodes.

    Returns:
        Float[Array, "nparams nparams"]: Hessian of ``loss_fn`` with respect to the
            dynamic parameter values.

    Examples:
        >>> import evermore as evm
        >>> import jax.numpy as jnp
        >>> params = {
        ...     "a": evm.Parameter(value=jnp.array([1.0])),
        ...     "b": evm.Parameter(value=jnp.array([2.0])),
        ... }
        >>> def loss_fn(pytree):
        ...     return jnp.sum(
        ...         (pytree["a"].value - 1.0) ** 2 + (pytree["b"].value - 2.0) ** 2
        ...     )
        >>> evm.loss.hessian_matrix(loss_fn, params).shape
        (2, 2)
    """
    graphdef, dynamic, rest = nnx.split(tree, is_dynamic_parameter, ...)
    flat_values, unravel_fn = _ravel_pure_tree(dynamic)

    def _flat_loss(flat_values: Float[Array, "..."]) -> Float[Array, ""]:
        param_values = unravel_fn(flat_values)

        # update the parameters with the new values
        # and call the loss function
        # 1. split the tree of parameters and other things
        # 2. update the parameters with the new values
        # 3. combine the updated parameters with the rest of the graph
        # 4. call the loss function with the updated tree

        # update them
        def _update(path, variable, value):
            del path  # unused
            return variable.replace(value=value)

        # using jax.tree.map here to not do inplace updates
        updated_dynamic = jax.tree.map_with_path(
            _update,
            dynamic,
            param_values,
            is_leaf=is_parameter,
            is_leaf_takes_path=True,
        )

        updated_tree = nnx.merge(graphdef, updated_dynamic, rest, copy=True)
        return loss_fn(updated_tree)

    # calculate hessian
    return jax.hessian(_flat_loss)(flat_values)


def fisher_information_matrix(
    loss_fn: tp.Callable,
    tree: PyTree[BaseParameter],
) -> Float[Array, "nparams nparams"]:
    """Builds the Fisher information matrix under the Laplace approximation.

    The Fisher matrix is obtained by evaluating the Hessian of ``loss_fn`` and
    inverting it. Only differentiable parameters, as determined by
    ``flax.nnx.split`` and ``evermore.filter.is_dynamic_parameter``, contribute.

    Args:
        loss_fn: Callable that accepts a PyTree of parameters and returns a scalar loss.
        tree: PyTree containing the parameters of interest.

    Returns:
        Float[Array, "nparams nparams"]: Fisher information matrix evaluated at
            the provided parameter values.

    Examples:
        >>> import evermore as evm
        >>> import jax.numpy as jnp
        >>> params = {
        ...     "a": evm.Parameter(value=jnp.array([1.0])),
        ...     "b": evm.Parameter(value=jnp.array([2.0])),
        ... }
        >>> def loss_fn(pytree):
        ...     return jnp.sum(
        ...         (pytree["a"].value - 1.0) ** 2 + (pytree["b"].value - 2.0) ** 2
        ...     )
        >>> evm.loss.fisher_information_matrix(loss_fn, params).shape
        (2, 2)
    """
    # calculate hessian
    hessian = hessian_matrix(loss_fn, tree)
    # invert to get the fisher information matrix under the Laplace assumption of normality
    return jnp.linalg.inv(hessian)


def covariance_matrix(
    loss_fn: tp.Callable,
    tree: PyTree[BaseParameter],
) -> Float[Array, "nparams nparams"]:
    """Derives a correlation matrix under the Laplace approximation.

    The Fisher information matrix is inverted and re-scaled so that the
    resulting matrix has unit diagonal entries. This corresponds to the
    correlation matrix implied by the Laplace approximation.

    Args:
        loss_fn: Callable that accepts a PyTree of parameters and returns a scalar loss.
        tree: PyTree containing the parameters of interest.

    Returns:
        Float[Array, "nparams nparams"]: Correlation matrix associated with the
            supplied parameter PyTree (diagonal entries are 1).

    Examples:
        >>> import evermore as evm
        >>> import jax.numpy as jnp
        >>> params = {
        ...     "a": evm.Parameter(value=jnp.array([1.0])),
        ...     "b": evm.Parameter(value=jnp.array([2.0])),
        ... }
        >>> def loss_fn(pytree):
        ...     return jnp.sum(
        ...         (pytree["a"].value - 1.0) ** 2 + (pytree["b"].value - 2.0) ** 2
        ...     )
        >>> evm.loss.covariance_matrix(loss_fn, params).shape
        (2, 2)
    """
    # calculate fisher information matrix
    fisher = fisher_information_matrix(loss_fn, tree)

    # normalize via D^-1 @ fisher @ D^-1 with D being the diagnonal standard deviation matrix
    d = jnp.sqrt(jnp.diagonal(fisher))
    cov = fisher / jnp.outer(d, d)

    # to avoid numerical issues, fix the diagonal to 1
    return jnp.where(jnp.eye(cov.shape[0], dtype=cov.dtype), 1.0, cov)


def cramer_rao_uncertainty(
    loss_fn: tp.Callable,
    tree: PyTree[BaseParameter],
) -> PyTree[BaseParameter]:
    """Estimates Cramér-Rao uncertainties under the Laplace approximation.

    The uncertainties are the square roots of the diagonal of the Fisher
    information matrix for the provided parameter PyTree.

    Args:
        loss_fn: Callable that accepts a PyTree of parameters and returns a scalar loss.
        tree: PyTree containing the parameters of interest.

    Returns:
        PyTree matching ``tree`` with each parameter replaced by its estimated
            standard deviation.

    Examples:
        >>> import evermore as evm
        >>> import jax.numpy as jnp
        >>> params = {
        ...     "a": evm.Parameter(value=jnp.array([1.0])),
        ...     "b": evm.Parameter(value=jnp.array([2.0])),
        ... }
        >>> def loss_fn(pytree):
        ...     return jnp.sum(
        ...         (pytree["a"].value - 1.0) ** 2 + (pytree["b"].value - 2.0) ** 2
        ...     )
        >>> uncertainties = evm.loss.cramer_rao_uncertainty(loss_fn, params)
        >>> {name: value.shape for name, value in uncertainties.items()}
        {'a': (1,), 'b': (1,)}
    """
    _, unravel_fn = _ravel_pure_tree(tree)

    # calculate fisher information matrix
    fisher_info = fisher_information_matrix(loss_fn, tree)
    return unravel_fn(jnp.sqrt(jnp.diag(fisher_info)))
