from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from evermore.custom_types import PDFLike
from evermore.parameter import Parameter
from evermore.util import _params_map

__all__ = [
    "get_log_probs",
    "get_boundary_constraints",
    "PoissonNLL",
]


def __dir__():
    return __all__


def get_log_probs(module: PyTree) -> PyTree:
    def _constraint(param: Parameter) -> Array:
        constraint = param.constraint
        if isinstance(constraint, PDFLike):
            constraint = cast(PDFLike, constraint)
            return constraint.log_prob(param.value)
        return jnp.array([0.0])

    # constraints from pdfs
    return _params_map(_constraint, module)


def get_boundary_constraints(module: PyTree) -> PyTree:
    return _params_map(lambda p: p.boundary_constraint, module)


class PoissonNLL(eqx.Module):
    """
    Poisson negative log-likelihood (NLL).

    Usage:

        .. code-block:: python

            import evermore as evm

            nll = evm.loss.PoissonNLL()

            def loss(model, x, y):
                expectation = model(x)
                loss = nll(expectation, y)
                constraints = evm.loss.get_log_probs(model)
                loss += evm.util.sum_leaves(constraints))
                return -jnp.sum(loss)
    """

    @property
    def logpdf(self) -> Callable:
        return jax.scipy.stats.poisson.logpmf

    @jax.named_scope("evm.loss.PoissonNLL")
    def __call__(self, expectation: Array, observation: Array) -> Array:
        # poisson log-likelihood
        return jnp.sum(
            self.logpdf(observation, expectation)
            - self.logpdf(observation, observation),
            axis=-1,
        )
