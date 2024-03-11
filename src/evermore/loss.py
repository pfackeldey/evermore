from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from evermore.custom_types import _NoValue
from evermore.parameter import Parameter
from evermore.pdf import PDF
from evermore.util import _params_map

__all__ = [
    "get_param_constraints",
    "PoissonNLL",
]


def __dir__():
    return __all__


def get_param_constraints(module: eqx.Module) -> dict:
    constraints = {}

    def _constraint(param: Parameter) -> Array:
        constraint = param.constraint
        if constraint is not _NoValue:
            constraint = cast(PDF, constraint)
            return constraint.logpdf(param.value)
        return jnp.array([0.0])

    # constraints from pdfs
    constraints["pdfs"] = _params_map(module, _constraint)
    # constraints from boundaries
    constraints["boundaries"] = _params_map(module, lambda p: p.boundary_penalty)
    return constraints


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
                constraints = evm.loss.get_param_constraints(model)
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
