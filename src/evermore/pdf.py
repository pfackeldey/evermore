from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln, xlogy
from jaxtyping import Array, PRNGKeyArray

from evermore.util import atleast_1d_float_array
from evermore.visualization import SupportsTreescope

__all__ = [
    "PDF",
    "Normal",
    "Poisson",
]


def __dir__():
    return __all__


class PDF(eqx.Module, SupportsTreescope):
    @abstractmethod
    def log_prob(self, x: Array) -> Array: ...

    @abstractmethod
    def sample(self, key: PRNGKeyArray) -> Array: ...

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

    def sample(self, key: PRNGKeyArray) -> Array:
        # sample parameter from pdf
        return self.mean + self.width * jax.random.normal(key)


class Poisson(PDF):
    lamb: Array = eqx.field(converter=atleast_1d_float_array)

    def log_prob(
        self, x: Array, normalize: bool = True, shift_mode: bool = False
    ) -> Array:
        # optionally adjust lambda to a higer value such that the new mode is the current lambda
        lamb = jnp.exp(digamma(self.lamb + 1)) if shift_mode else self.lamb

        def _continous_poisson_log_prob(x, lamb):
            x = jnp.array(x, jnp.result_type(float))
            return xlogy(x, lamb) - lamb - gammaln(x + 1)

        unnormalized = _continous_poisson_log_prob(x, lamb)

        if normalize:
            args = (self.lamb, lamb) if shift_mode else (x, x)
            logpdf_max = _continous_poisson_log_prob(*args)
            return unnormalized - logpdf_max

        return unnormalized

    def sample(self, key: PRNGKeyArray) -> Array:
        # this samples only integers, do we want that?
        return jax.random.poisson(key, self.lamb)
