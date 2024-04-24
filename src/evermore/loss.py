from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from evermore.custom_types import PDFLike
from evermore.parameter import Parameter, params_map

__all__ = [
    "get_log_probs",
    "get_boundary_constraints",
    "PoissonLogLikelihood",
]


def __dir__():
    return __all__


def get_log_probs(module: PyTree) -> PyTree:
    def _constraint(param: Parameter) -> Array:
        prior = param.prior
        if isinstance(prior, PDFLike):
            prior = cast(PDFLike, prior)
            return prior.log_prob(param.value)
        return jnp.array([0.0])

    # constraints from pdfs
    return params_map(_constraint, module)


def get_boundary_constraints(module: PyTree) -> PyTree:
    return params_map(lambda p: p.boundary_constraint, module)


class PoissonLogLikelihood(eqx.Module):
    """
    Poisson log-likelihood.

    Usage:

    .. code-block:: python

        import evermore as evm

        nll = evm.loss.PoissonLogLikelihood()

        def loss(model, x, y):
            expectation = model(x)
            loss = nll(expectation, y)
            constraints = evm.loss.get_log_probs(model)
            loss += evm.util.sum_over_leaves(constraints)
            return -jnp.sum(loss)
    """

    @property
    def log_prob(self) -> Callable:
        return jax.scipy.stats.poisson.logpmf

    @jax.named_scope("evm.loss.PoissonLogLikelihood")
    def __call__(self, expectation: Array, observation: Array) -> Array:
        # poisson log-likelihood
        return jnp.sum(
            self.log_prob(observation, expectation)
            - self.log_prob(observation, observation),
            axis=-1,
        )
