from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

if TYPE_CHECKING:
    from evermore import Parameter

__all__ = [
    "HashablePDF",
    "Flat",
    "Gauss",
    "Poisson",
]


def __dir__():
    return __all__


class HashablePDF(eqx.Module):
    @abstractmethod
    def __hash__(self) -> int:
        ...

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
    def inv_cdf(self, x: Array) -> Array:
        ...

    @abstractmethod
    def sample(self, key: PRNGKeyArray, parameter: Parameter) -> Array:
        ...


class Flat(HashablePDF):
    def __hash__(self):
        return hash(self.__class__)

    def logpdf(self, x: Array) -> Array:
        return jnp.array([0.0])

    def pdf(self, x: Array) -> Array:
        return jnp.array([1.0])

    def cdf(self, x: Array) -> Array:
        return jnp.array([1.0])

    def inv_cdf(self, x: Array) -> Array:
        msg = "Flat distribution has no inverse CDF."
        raise ValueError(msg)

    def sample(self, key: PRNGKeyArray, parameter: Parameter) -> Array:
        return jax.random.uniform(
            key,
            parameter.value.shape,
            # what should be the ranges?
            # +/-jnp.inf leads to nans...
            # minval=parameter.lower,
            # maxval=parameter.upper,
        )


class Gauss(HashablePDF):
    mean: float = eqx.field(static=True)
    width: float = eqx.field(static=True)

    def __init__(self, mean: float, width: float) -> None:
        self.mean = mean
        self.width = width

    def __hash__(self):
        return hash(self.__class__) ^ hash((self.mean, self.width))

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

    def inv_cdf(self, x: Array) -> Array:
        return jax.scipy.stats.norm.ppf(x, loc=self.mean, scale=self.width)

    def sample(self, key: PRNGKeyArray, parameter: Parameter) -> Array:
        return self.mean + self.width * jax.random.normal(
            key,
            shape=parameter.value.shape,
            dtype=parameter.value.dtype,
        )


class Poisson(HashablePDF):
    lamb: Array = eqx.field(static=True)

    def __init__(self, lamb: Array) -> None:
        self.lamb = lamb

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other: Any):  # type: ignore[override]
        if not isinstance(other, Poisson):
            return ValueError(f"Cannot compare Poisson with {type(other)}")
        # We need to implement __eq__ explicitely because we have a non-hashable field (lamb).
        # Implementing __eq__ is necessary for the `==` operator to work and to ensure that the
        # Poisson distribution is correctly added to a python set.
        return jnp.all(self.lamb == other.lamb)

    def logpdf(self, x: Array) -> Array:
        logpdf_max = jax.scipy.stats.poisson.logpmf(self.lamb, mu=self.lamb)
        unnormalized = jax.scipy.stats.poisson.logpmf((x + 1) * self.lamb, mu=self.lamb)
        return unnormalized - logpdf_max

    def pdf(self, x: Array) -> Array:
        return jax.scipy.stats.poisson.pmf((x + 1) * self.lamb, mu=self.lamb)

    def cdf(self, x: Array) -> Array:
        return jax.scipy.stats.poisson.cdf((x + 1) * self.lamb, mu=self.lamb)

    def inv_cdf(self, x: Array) -> Array:
        # see: https://num.pyro.ai/en/stable/tutorials/truncated_distributions.html?highlight=poisson%20inverse#5.3-Example:-Left-truncated-Poisson
        def cond_fn(val):
            n, cdf = val
            return jnp.any(cdf < x)

        def body_fn(val):
            n, cdf = val
            n_new = jnp.where(cdf < x, n + 1, n)
            return n_new, self.cdf(n_new)

        start = jnp.zeros_like(x)
        cdf_start = self.cdf(start)
        n, _ = jax.lax.while_loop(cond_fn, body_fn, (start, cdf_start))
        return n.astype(jnp.result_type(int))

    def sample(self, key: PRNGKeyArray) -> Array:  # type: ignore[override]
        return jax.random.poisson(
            key,
            self.lamb,
            shape=self.lamb.shape,
            dtype=self.lamb.dtype,
        )
