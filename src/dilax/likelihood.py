from __future__ import annotations

from typing import TYPE_CHECKING, cast

import equinox as eqx
import jax
import jax.numpy as jnp

from dilax.model import Model
from dilax.util import Sentinel, _NoValue


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

    def __call__(self, values: dict[str, jax.Array] | Sentinel = _NoValue) -> jax.Array:
        if values is _NoValue:
            values = {}
        model = self.model.update(values=values)
        res = model.evaluate()
        nll = (
            self.logpdf(self.observation, res.expectation())
            - self.logpdf(self.observation, self.observation)
            + model.nll_boundary_penalty()
            + model.parameter_constraints()
        )
        return -jnp.sum(nll, axis=-1)


class Hessian(BaseModule):
    """
    Hessian matrix.
    """

    NLL: NLL

    def __init__(self, model: Model, observation: jax.Array) -> None:
        super().__init__(model=model, observation=observation)
        self.NLL = NLL(model=model, observation=observation)

    def __call__(self, values: dict[str, jax.Array] | Sentinel = _NoValue) -> jax.Array:
        if values is _NoValue:
            values = {}
        if not values:
            values = self.model.parameter_values
        if TYPE_CHECKING:
            values = cast(dict[str, jax.Array], values)
        hessian = jax.hessian(self.NLL, argnums=0)(values)
        hessian, _ = jax.tree_util.tree_flatten(hessian)
        hessian = jnp.array(hessian)
        new_shape = len(values)
        return jnp.reshape(hessian, (new_shape, new_shape))


class CovMatrix(Hessian):
    """
    Covariance matrix.
    """

    def __call__(self, values: dict[str, jax.Array] | Sentinel = _NoValue) -> jax.Array:
        if values is _NoValue:
            values = {}
        hessian = super().__call__(values=values)
        return jnp.linalg.inv(-hessian)


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
        values: dict[str, jax.Array] | Sentinel = _NoValue,
        key: jax.Array | Sentinel = _NoValue,
    ) -> dict[str, jax.Array]:
        if values is _NoValue:
            values = {}
        if key is _NoValue:
            key = jax.random.PRNGKey(1234)
        if TYPE_CHECKING:
            key = cast(jax.Array, key)
        if not values:
            values = self.model.parameter_values
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
