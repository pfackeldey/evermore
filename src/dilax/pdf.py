from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp


class HashablePDF(eqx.Module):
    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def logpdf(self, x: jax.Array) -> jax.Array:
        ...

    @abstractmethod
    def pdf(self, x: jax.Array) -> jax.Array:
        ...

    @abstractmethod
    def cdf(self, x: jax.Array) -> jax.Array:
        ...

    @abstractmethod
    def inv_cdf(self, x: jax.Array) -> jax.Array:
        ...


class Flat(HashablePDF):
    def __hash__(self):
        return hash(self.__class__)

    def logpdf(self, x: jax.Array) -> jax.Array:
        return jnp.array([0.0])

    def pdf(self, x: jax.Array) -> jax.Array:
        return jnp.array([1.0])

    def cdf(self, x: jax.Array) -> jax.Array:
        return jnp.array([1.0])

    def inv_cdf(self, x: jax.Array) -> jax.Array:
        msg = "Flat distribution has no inverse CDF."
        raise ValueError(msg)


class Gauss(HashablePDF):
    mean: float = eqx.field(static=True)
    width: float = eqx.field(static=True)

    def __init__(self, mean: float, width: float) -> None:
        self.mean = mean
        self.width = width

    def __hash__(self):
        return hash(self.__class__) ^ hash((self.mean, self.width))

    def logpdf(self, x: jax.Array) -> jax.Array:
        logpdf_max = jax.scipy.stats.norm.logpdf(
            self.mean, loc=self.mean, scale=self.width
        )
        unnormalized = jax.scipy.stats.norm.logpdf(x, loc=self.mean, scale=self.width)
        return unnormalized - logpdf_max

    def pdf(self, x: jax.Array) -> jax.Array:
        return jax.scipy.stats.norm.pdf(x, loc=self.mean, scale=self.width)

    def cdf(self, x: jax.Array) -> jax.Array:
        return jax.scipy.stats.norm.cdf(x, loc=self.mean, scale=self.width)

    def inv_cdf(self, x: jax.Array) -> jax.Array:
        return jax.scipy.stats.norm.ppf(x, loc=self.mean, scale=self.width)


class Poisson(HashablePDF):
    lamb: int = eqx.field(static=True)

    def __init__(self, lamb: int) -> None:
        self.lamb = lamb

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.lamb)

    def logpdf(self, x: jax.Array) -> jax.Array:
        logpdf_max = jax.scipy.stats.poisson.logpmf(self.lamb, mu=self.lamb)
        unnormalized = jax.scipy.stats.poisson.logpmf(x, mu=self.lamb)
        return unnormalized - logpdf_max

    def pdf(self, x: jax.Array) -> jax.Array:
        return jax.scipy.stats.poisson.pmf(x, mu=self.lamb)

    def cdf(self, x: jax.Array) -> jax.Array:
        return jax.scipy.stats.poisson.cdf(x, mu=self.lamb)

    def inv_cdf(self, x: jax.Array) -> jax.Array:
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
