from typing import TYPE_CHECKING, cast

import equinox as eqx
import jax
import jax.numpy as jnp

from dilax.custom_types import Sentinel, _NoValue
from dilax.model import Model

__all__ = [
    "NLL",
    "Hessian",
    "CovMatrix",
    "SampleToy",
]


def __dir__():
    return __all__


class BaseModule(eqx.Module):
    """
    Base module to hold the `model` and the `observation`.
    """

    model: Model
    observation: jax.Array = eqx.field(converter=jnp.asarray)

    def __init__(self, model: Model, observation: jax.Array) -> None:
        self.model = model
        self.observation = observation


class NLL(BaseModule):
    """
    Negative log-likelihood (NLL).
    """

    def logpdf(self, *args, **kwargs) -> jax.Array:
        return jax.scipy.stats.poisson.logpmf(*args, **kwargs)

    def __call__(self, values: dict | Sentinel = _NoValue) -> jax.Array:
        if values is _NoValue:
            values = self.model.parameter_values
        model = self.model.update(values=values)
        res = model.evaluate()
        nll = jnp.sum(
            self.logpdf(self.observation, res.expectation())
            - self.logpdf(self.observation, self.observation),
            axis=-1,
        )
        # add constraints
        constraints = jax.tree_util.tree_leaves(model.parameter_constraints())
        nll += sum(constraints)
        nll += model.nll_boundary_penalty()
        return -jnp.sum(nll)


class Hessian(BaseModule):
    """
    Hessian matrix.
    """

    NLL: NLL

    def __init__(self, model: Model, observation: jax.Array) -> None:
        super().__init__(model=model, observation=observation)
        self.NLL = NLL(model=model, observation=observation)

    def __call__(self, values: dict | Sentinel = _NoValue) -> jax.Array:
        if values is _NoValue:
            values = self.model.parameter_values
        if TYPE_CHECKING:
            values = cast(dict, values)
        hessian = jax.hessian(self.NLL, argnums=0)(values)
        hessian, _ = jax.tree_util.tree_flatten(hessian)
        hessian = jnp.array(hessian)
        new_shape = len(values)
        return jnp.reshape(hessian, (new_shape, new_shape))


class CovMatrix(Hessian):
    """
    Covariance matrix.
    """

    def __call__(self, values: dict | Sentinel = _NoValue) -> jax.Array:
        if values is _NoValue:
            values = self.model.parameter_values
        hessian = super().__call__(values=values)
        return jnp.linalg.inv(hessian)


class SampleToy(BaseModule):
    """
    Sample a toy from the model.
    """

    CovMatrix: CovMatrix

    def __init__(self, model: Model, observation: jax.Array) -> None:
        super().__init__(model=model, observation=observation)
        self.CovMatrix = CovMatrix(model=model, observation=observation)

    def __call__(
        self,
        values: dict | Sentinel = _NoValue,
        key: jax.Array | Sentinel = _NoValue,
    ) -> dict[str, jax.Array]:
        if values is _NoValue:
            values = self.model.parameter_values
        if key is _NoValue:
            key = jax.random.PRNGKey(1234)
        if TYPE_CHECKING:
            key = cast(jax.Array, key)
        cov = self.CovMatrix(values=values)
        _values, tree_def = jax.tree_util.tree_flatten(
            self.model.update(values=values).parameter_values
        )
        sampled_values = jax.random.multivariate_normal(
            key=key,
            mean=jnp.concatenate(_values),
            cov=cov,
        )
        new_values = jax.tree_util.tree_unflatten(tree_def, sampled_values)
        model = self.model.update(values=new_values)
        return model.evaluate().expectations
