from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

if TYPE_CHECKING:
    pass

__all__ = [
    "PDF",
    "Flat",
    "Gauss",
    "Poisson",
]


def __dir__():
    return __all__


class PDF(eqx.Module):
    @abstractmethod
    def logpdf(self, x: Array) -> Array:
        ...

    @abstractmethod
    def pdf(self, x: Array) -> Array:
        ...

    @abstractmethod
    def cdf(self, x: Array) -> Array:
        ...

    @abstractmethod
    def sample(self, key: PRNGKeyArray) -> Array:
        ...


class Flat(PDF):
    def logpdf(self, x: Array) -> Array:
        return jnp.zeros_like(x)

    def pdf(self, x: Array) -> Array:
        return jnp.ones_like(x)

    def cdf(self, x: Array) -> Array:
        return jnp.ones_like(x)

    def sample(self, key: PRNGKeyArray) -> Array:
        # sample parameter from pdf
        # what should be the ranges?
        # +/-jnp.inf leads to nans...
        # minval=??,
        # maxval=??,
        return jax.random.uniform(key)


class Gauss(PDF):
    mean: Array
    width: Array

    def logpdf(self, x: Array) -> Array:
        logpdf_max = jax.scipy.stats.norm.logpdf(
            self.mean, loc=self.mean, scale=self.width
        )
        unnormalized = jax.scipy.stats.norm.logpdf(x, loc=self.mean, scale=self.width)
        return unnormalized - logpdf_max

    def pdf(self, x: Array) -> Array:
        return jax.scipy.stats.norm.pdf(x, loc=self.mean, scale=self.width)

    def cdf(self, x: Array) -> Array:
        return jax.scipy.stats.norm.cdf(x, loc=self.mean, scale=self.width)

    def sample(self, key: PRNGKeyArray) -> Array:
        # sample parameter from pdf
        return self.mean + self.width * jax.random.normal(key)


class Poisson(PDF):
    lamb: Array

    def logpdf(self, x: Array) -> Array:
        logpdf_max = jax.scipy.stats.poisson.logpmf(self.lamb, mu=self.lamb)
        unnormalized = jax.scipy.stats.poisson.logpmf((x + 1) * self.lamb, mu=self.lamb)
        return unnormalized - logpdf_max

    def pdf(self, x: Array) -> Array:
        return jax.scipy.stats.poisson.pmf((x + 1) * self.lamb, mu=self.lamb)

    def cdf(self, x: Array) -> Array:
        return jax.scipy.stats.poisson.cdf((x + 1) * self.lamb, mu=self.lamb)

    def sample(self, key: PRNGKeyArray) -> Array:
        # sample parameter from pdf
        # some problems with this:
        #  - this samples only integers, do we want that?
        #  - this breaks for 0 in self.lamb
        #  - if jax.random.poisson(key, self.lamb) == 0 then what do we know about the parameter?
        return (jax.random.poisson(key, self.lamb) / self.lamb) - 1
