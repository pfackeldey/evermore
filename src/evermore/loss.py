from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from evermore.parameter import Parameter
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
        if param.constraints:
            if len(param.constraints) > 1:
                msg = f"More than one constraint per parameter is not allowed. Got: {param.constraints}"
                raise ValueError(msg)
            return next(iter(param.constraints)).logpdf(param.value)
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
                constraints = evm.loss.get_param_constraints(model)
                loss = nll(expectation, y, evm.util.sum_leaves(constraints))
                return loss
    """

    @property
    def logpdf(self) -> Callable:
        return jax.scipy.stats.poisson.logpmf

    @jax.named_scope("evm.loss.PoissonNLL")
    def __call__(
        self, expectation: Array, observation: Array, constraint: Array
    ) -> Array:
        # poisson log-likelihood
        nll = jnp.sum(
            self.logpdf(observation, expectation)
            - self.logpdf(observation, observation),
            axis=-1,
        )
        # add constraint
        nll += constraint
        return -jnp.sum(nll)
