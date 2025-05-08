from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jax._src.random import Shape
from jax.scipy.special import digamma, gammaln, xlogy
from jaxtyping import Array, PRNGKeyArray

from evermore.util import atleast_1d_float_array
from evermore.visualization import SupportsTreescope

__all__ = [
    "PDF",
    "Normal",
    "PoissonBase",
    "PoissonContinuous",
    "PoissonDiscrete",
]


def __dir__():
    return __all__


class PDF(eqx.Module, SupportsTreescope):
    @abc.abstractmethod
    def log_prob(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def param_to_pdf(self, value: Array) -> Array: ...

    @abc.abstractmethod
    def pdf_to_param(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def sample(self, key: PRNGKeyArray, shape: Shape | None = None) -> Array: ...

    def prob(self, x: Array, **kwargs) -> Array:
        return jnp.exp(self.log_prob(x, **kwargs))


class Normal(PDF):
    mean: Array = eqx.field(converter=atleast_1d_float_array)
    width: Array = eqx.field(converter=atleast_1d_float_array)

    def log_prob(self, x: Array) -> Array:
        logpdf_max = jax.scipy.stats.norm.logpdf(
            self.mean, loc=self.mean, scale=self.width
        )
        unnormalized = jax.scipy.stats.norm.logpdf(x, loc=self.mean, scale=self.width)
        return unnormalized - logpdf_max

    def param_to_pdf(self, value: Array) -> Array:
        # normal scaling via mean and width
        return self.mean + self.width * value

    def pdf_to_param(self, x: Array) -> Array:
        return (x - self.mean) / self.width

    def sample(self, key: PRNGKeyArray, shape: Shape | None = None) -> Array:
        # jax.random.normal does not accept None shape
        if shape is None:
            shape = ()
        # sample parameter from pdf
        return self.param_to_pdf(jax.random.normal(key, shape=shape))


class PoissonBase(PDF):
    lamb: Array = eqx.field(converter=atleast_1d_float_array)


class PoissonDiscrete(PoissonBase):
    """
    Poisson distribution with discrete support. Float inputs are floored to the nearest integer.
    See https://root.cern.ch/doc/master/RooPoisson_8cxx_source.html#l00057 for reference.
    """

    def log_prob(self, x: Array, normalize: bool = True) -> Array:
        x = jnp.floor(x)

        unnormalized = jax.scipy.stats.poisson.logpmf(x, self.lamb)
        if not normalize:
            return unnormalized

        logpdf_max = jax.scipy.stats.poisson.logpmf(x, x)
        return unnormalized - logpdf_max

    def param_to_pdf(self, value: Array) -> Array:
        # convert the value to a normal cdf value to look for
        target_cdf = jax.scipy.stats.norm.cdf(value)

        # perform an iterative search
        # see: https://num.pyro.ai/en/stable/tutorials/truncated_distributions.html?highlight=poisson%20inverse#5.3-Example:-Left-truncated-Poisson
        def cond_fn(val):
            n, cdf = val
            return jnp.any(cdf < target_cdf)

        def body_fn(val):
            n, cdf = val
            n_new = jnp.where(cdf < target_cdf, n + 1, n)
            return n_new, jax.scipy.stats.poisson.cdf(n_new, self.lamb)

        start_n = jnp.zeros_like(value, dtype=jnp.result_type(int))
        start_cdf = jnp.zeros_like(value, dtype=jnp.result_type(float))
        n, _ = jax.lax.while_loop(cond_fn, body_fn, (start_n, start_cdf))

        # since we check for cdf < value, n will always refer to the next value
        return jnp.clip(n - 1, min=0)

    def pdf_to_param(self, x: Array) -> Array:
        x = jnp.floor(x)
        cdf = jax.scipy.stats.poisson.cdf(x, self.lamb)
        return jax.scipy.stats.norm.ppf(cdf)

    def sample(self, key: PRNGKeyArray, shape: Shape | None = None) -> Array:
        # jax.random.poisson does not accept empty tuple shape
        if shape == ():
            shape = None
        return jax.random.poisson(key, self.lamb, shape=shape)


class PoissonContinuous(PoissonBase):
    def log_prob(
        self, x: Array, normalize: bool = True, shift_mode: bool = False
    ) -> Array:
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

    def param_to_pdf(self, value: Array) -> Array:
        err = f"{self.__class__.__name__} does not support param_to_pdf"
        raise Exception(err)

    def pdf_to_param(self, x: Array) -> Array:
        err = f"{self.__class__.__name__} does not support pdf_to_param"
        raise Exception(err)

    def sample(self, key: PRNGKeyArray, shape: Shape | None = None) -> Array:
        msg = f"{self.__class__.__name__} does not support sampling, use PoissonDiscrete instead"
        raise Exception(msg)
