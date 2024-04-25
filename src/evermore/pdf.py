from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

__all__ = [
    "PDF",
    "Normal",
    "Poisson",
]


def __dir__():
    return __all__


class PDF(eqx.Module):
    @abstractmethod
    def log_prob(self, x: Array) -> Array: ...

    @abstractmethod
    def sample(self, key: PRNGKeyArray) -> Array: ...


class Normal(PDF):
    mean: Array = eqx.field(converter=jnp.atleast_1d)
    width: Array = eqx.field(converter=jnp.atleast_1d)

    def log_prob(self, x: Array) -> Array:
        logpdf_max = jax.scipy.stats.norm.logpdf(
            self.mean, loc=self.mean, scale=self.width
        )
        unnormalized = jax.scipy.stats.norm.logpdf(x, loc=self.mean, scale=self.width)
        return unnormalized - logpdf_max

    def sample(self, key: PRNGKeyArray) -> Array:
        # sample parameter from pdf
        return self.mean + self.width * jax.random.normal(key)


class Poisson(PDF):
    lamb: Array = eqx.field(converter=jnp.atleast_1d)

    def log_prob(self, x: Array) -> Array:
        logpdf_max = jax.scipy.stats.poisson.logpmf(self.lamb, mu=self.lamb)
        unnormalized = jax.scipy.stats.poisson.logpmf((x + 1) * self.lamb, mu=self.lamb)
        return unnormalized - logpdf_max

    def sample(self, key: PRNGKeyArray) -> Array:
        # this samples only integers, do we want that?
        return jax.random.poisson(key, self.lamb)
