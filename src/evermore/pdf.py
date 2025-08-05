from __future__ import annotations

import abc
from typing import Generic, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax._src.random import Shape
from jax.scipy.special import digamma, gammaln, xlogy
from jaxtyping import Array, Float, PRNGKeyArray

from evermore.parameters.parameter import V
from evermore.util import maybe_float_array
from evermore.visualization import SupportsTreescope

__all__ = [
    "AbstractPDF",
    "Normal",
    "PoissonBase",
    "PoissonContinuous",
    "PoissonDiscrete",
]


def __dir__():
    return __all__


@runtime_checkable
class ImplementsFromUnitNormalConversion(Protocol[V]):
    def __evermore_from_unit_normal__(self, x: V) -> V: ...


class AbstractPDF(eqx.Module, Generic[V], SupportsTreescope):
    @abc.abstractmethod
    def log_prob(self, x: V) -> V: ...

    @abc.abstractmethod
    def cdf(self, x: V) -> V: ...

    @abc.abstractmethod
    def inv_cdf(self, x: V) -> V: ...

    @abc.abstractmethod
    def sample(self, key: PRNGKeyArray, shape: Shape) -> Float[Array, ...]: ...

    def prob(self, x: V, **kwargs) -> V:
        return jnp.exp(self.log_prob(x, **kwargs))


class Normal(AbstractPDF[V]):
    mean: V = eqx.field(converter=maybe_float_array)
    width: V = eqx.field(converter=maybe_float_array)

    def log_prob(self, x: V) -> V:
        logpdf_max = jax.scipy.stats.norm.logpdf(
            self.mean, loc=self.mean, scale=self.width
        )
        unnormalized = jax.scipy.stats.norm.logpdf(x, loc=self.mean, scale=self.width)
        return unnormalized - logpdf_max

    def cdf(self, x: V) -> V:
        return jax.scipy.stats.norm.cdf(x, loc=self.mean, scale=self.width)

    def inv_cdf(self, x: V) -> V:
        return jax.scipy.stats.norm.ppf(x, loc=self.mean, scale=self.width)

    def __evermore_from_unit_normal__(self, x: V) -> V:
        return self.mean + self.width * x

    def sample(self, key: PRNGKeyArray, shape: Shape) -> Float[Array, ...]:
        # sample parameter from pdf
        return self.__evermore_from_unit_normal__(jax.random.normal(key, shape=shape))


class PoissonBase(AbstractPDF[V]):
    lamb: V = eqx.field(converter=maybe_float_array)


class PoissonDiscrete(PoissonBase[V]):
    """
    Poisson distribution with discrete support. Float inputs are floored to the nearest integer.
    See https://root.cern.ch/doc/master/RooPoisson_8cxx_source.html#l00057 for reference.
    """

    def log_prob(
        self,
        x: V,
        normalize: bool = True,
    ) -> V:
        x = jnp.floor(x)

        unnormalized = jax.scipy.stats.poisson.logpmf(x, self.lamb)
        if not normalize:
            return unnormalized

        logpdf_max = jax.scipy.stats.poisson.logpmf(x, x)
        return unnormalized - logpdf_max

    def cdf(self, x: V) -> V:
        return jax.scipy.stats.poisson.cdf(x, self.lamb)

    def inv_cdf(self, x: V) -> V:
        # perform an iterative search
        # see: https://num.pyro.ai/en/stable/tutorials/truncated_distributions.html?highlight=poisson%20inverse#5.3-Example:-Left-truncated-Poisson
        def cond_fn(val):
            _, cdf = val
            return jnp.any(cdf < x)

        def body_fn(val):
            n, cdf = val
            n_new = jnp.where(cdf < x, n + 1, n)
            return n_new, jax.scipy.stats.poisson.cdf(n_new, self.lamb)

        start_n = jnp.zeros_like(x, dtype=jnp.result_type(int))
        start_cdf = jnp.zeros_like(x, dtype=jnp.result_type(float))
        n, _ = jax.lax.while_loop(cond_fn, body_fn, (start_n, start_cdf))

        # since we check for cdf < value, n will always refer to the next value
        return jnp.clip(n - 1, min=0)

    def sample(self, key: PRNGKeyArray, shape: Shape) -> Float[Array, ...]:
        return jax.random.poisson(key, self.lamb, shape=shape)


class PoissonContinuous(PoissonBase[V]):
    def log_prob(
        self,
        x: V,
        normalize: bool = True,
        shift_mode: bool = False,
    ) -> V:
        # optionally adjust lambda to a higher value such that the new mode is the current lambda
        lamb = jnp.exp(digamma(self.lamb + 1)) if shift_mode else self.lamb

        def _log_prob(x, lamb):
            x = jnp.array(x, jnp.result_type(float))
            return xlogy(x, lamb) - lamb - gammaln(x + 1)

        unnormalized = _log_prob(x, lamb)
        if not normalize:
            return unnormalized

        args = (self.lamb, lamb) if shift_mode else (x, x)
        logpdf_max = _log_prob(*args)
        return unnormalized - logpdf_max

    def cdf(self, x: V) -> V:
        err = f"{self.__class__.__name__} does not support cdf"
        raise Exception(err)

    def inv_cdf(self, x: V) -> V:
        err = f"{self.__class__.__name__} does not support inv_cdf"
        raise Exception(err)

    def sample(
        self, key: PRNGKeyArray, shape: Shape | None = None
    ) -> Float[Array, ...]:
        msg = f"{self.__class__.__name__} does not support sampling, use PoissonDiscrete instead"
        raise Exception(msg)
