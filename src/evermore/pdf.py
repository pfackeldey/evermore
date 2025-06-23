from __future__ import annotations

import abc
import typing as tp

import equinox as eqx
import jax
import jax.numpy as jnp
from jax._src.random import Shape
from jax.scipy.special import digamma, gammaln, xlogy
from jaxtyping import Array, Float, PRNGKeyArray

from evermore.util import float_array
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


@tp.runtime_checkable
class ImplementsFromUnitNormalConversion(tp.Protocol):
    def __evermore_from_unit_normal__(
        self,
        x: Float[Array, "..."],  # noqa: UP037
    ) -> Float[Array, "..."]: ...  # noqa: UP037


class PDF(eqx.Module, SupportsTreescope):
    @abc.abstractmethod
    def log_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]: ...  # noqa: UP037

    @abc.abstractmethod
    def cdf(self, x: Float[Array, "..."]) -> Float[Array, "..."]: ...  # noqa: UP037

    @abc.abstractmethod
    def inv_cdf(self, x: Float[Array, "..."]) -> Float[Array, "..."]: ...  # noqa: UP037

    @abc.abstractmethod
    def sample(
        self, key: PRNGKeyArray, shape: Shape | None = None
    ) -> Float[Array, "..."]: ...  # noqa: UP037

    def prob(self, x: Float[Array, "..."], **kwargs) -> Float[Array, "..."]:  # noqa: UP037
        return jnp.exp(self.log_prob(x, **kwargs))


class Normal(PDF):
    mean: Float[Array, "..."] = eqx.field(converter=float_array)  # noqa: UP037
    width: Float[Array, "..."] = eqx.field(converter=float_array)  # noqa: UP037

    def log_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:  # noqa: UP037
        logpdf_max = jax.scipy.stats.norm.logpdf(
            self.mean, loc=self.mean, scale=self.width
        )
        unnormalized = jax.scipy.stats.norm.logpdf(x, loc=self.mean, scale=self.width)
        return unnormalized - logpdf_max

    def cdf(self, x: Float[Array, "..."]) -> Float[Array, "..."]:  # noqa: UP037
        return jax.scipy.stats.norm.cdf(x, loc=self.mean, scale=self.width)

    def inv_cdf(self, x: Float[Array, "..."]) -> Float[Array, "..."]:  # noqa: UP037
        return jax.scipy.stats.norm.ppf(x, loc=self.mean, scale=self.width)

    def __evermore_from_unit_normal__(self, x: Array) -> Array:
        return self.mean + self.width * x

    def sample(
        self, key: PRNGKeyArray, shape: Shape | None = None
    ) -> Float[Array, "..."]:  # noqa: UP037
        # jax.random.normal does not accept None shape
        if shape is None:
            shape = ()
        # sample parameter from pdf
        return self.__evermore_from_unit_normal__(jax.random.normal(key, shape=shape))


class PoissonBase(PDF):
    lamb: Float[Array, "..."] = eqx.field(converter=float_array)  # noqa: UP037


class PoissonDiscrete(PoissonBase):
    """
    Poisson distribution with discrete support. Float inputs are floored to the nearest integer.
    See https://root.cern.ch/doc/master/RooPoisson_8cxx_source.html#l00057 for reference.
    """

    def log_prob(
        self,
        x: Float[Array, "..."],  # noqa: UP037
        normalize: bool = True,
    ) -> Float[Array, "..."]:  # noqa: UP037
        x = jnp.floor(x)

        unnormalized = jax.scipy.stats.poisson.logpmf(x, self.lamb)
        if not normalize:
            return unnormalized

        logpdf_max = jax.scipy.stats.poisson.logpmf(x, x)
        return unnormalized - logpdf_max

    def cdf(self, x: Float[Array, "..."]) -> Float[Array, "..."]:  # noqa: UP037
        return jax.scipy.stats.poisson.cdf(x, self.lamb)

    def inv_cdf(self, x: Float[Array, "..."]) -> Float[Array, "..."]:  # noqa: UP037
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

    def sample(
        self, key: PRNGKeyArray, shape: Shape | None = None
    ) -> Float[Array, "..."]:  # noqa: UP037
        # jax.random.poisson does not accept empty tuple shape
        if shape == ():
            shape = None
        return jax.random.poisson(key, self.lamb, shape=shape)


class PoissonContinuous(PoissonBase):
    def log_prob(
        self,
        x: Float[Array, "..."],  # noqa: UP037
        normalize: bool = True,
        shift_mode: bool = False,
    ) -> Float[Array, "..."]:  # noqa: UP037
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

    def cdf(self, x: Float[Array, "..."]) -> Float[Array, "..."]:  # noqa: UP037
        err = f"{self.__class__.__name__} does not support cdf"
        raise Exception(err)

    def inv_cdf(self, x: Float[Array, "..."]) -> Float[Array, "..."]:  # noqa: UP037
        err = f"{self.__class__.__name__} does not support inv_cdf"
        raise Exception(err)

    def sample(
        self, key: PRNGKeyArray, shape: Shape | None = None
    ) -> Float[Array, "..."]:  # noqa: UP037
        msg = f"{self.__class__.__name__} does not support sampling, use PoissonDiscrete instead"
        raise Exception(msg)
