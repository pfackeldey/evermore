from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln, xlogy
from jax._src.random import Shape
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
    def scale_std(self, value: Array) -> Array: ...

    @abstractmethod
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

    def scale_std(self, value: Array) -> Array:
        # normal scaling via mean and width
        return self.mean + self.width * value

    def sample(self, key: PRNGKeyArray, shape: Shape | None = None) -> Array:
        # jax.random.normal does not accept None shape
        if shape is None:
            shape = ()
        # sample parameter from pdf
        return self.scale_std(jax.random.normal(key, shape=shape))


class Poisson(PDF):
    lamb: Array = eqx.field(converter=atleast_1d_float_array)

    def log_prob(
        self, x: Array, normalize: bool = True, shift_mode: bool = False
    ) -> Array:
        # optionally adjust lambda to a higher value such that the new mode is the current lambda
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

    def scale_std(self, value: Array) -> Array:
        err = f"{self.__class__.__name__} does not support scale_std"
        raise Exception(err)

    def sample(self, key: PRNGKeyArray, shape: Shape | None = None) -> Array:
        # jax.random.poisson does not accept empty tuple shape
        if shape == ():
            shape = None
        # this samples only integers, do we want that?
        return jax.random.poisson(key, self.lamb, shape=shape)
