from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx
from jax._src.random import Shape
from jax.scipy.special import digamma, gammaln, xlogy
from jaxtyping import Array, Float, PRNGKeyArray

from evermore.parameters.parameter import V
from evermore.util import float_array

__all__ = [
    "BasePDF",
    "Normal",
    "PoissonBase",
    "PoissonContinuous",
    "PoissonDiscrete",
]


def __dir__():
    return __all__


@runtime_checkable
class ImplementsFromUnitNormalConversion(Protocol):
    def __evermore_from_unit_normal__(self, x: V) -> V: ...


class BasePDF(nnx.Pytree):
    @abc.abstractmethod
    def log_prob(self, x: V) -> V: ...

    @abc.abstractmethod
    def cdf(self, x: V) -> V: ...

    @abc.abstractmethod
    def inv_cdf(self, x: V) -> V: ...

    @abc.abstractmethod
    def sample(self, key: PRNGKeyArray, shape: Shape) -> Float[Array, ...]: ...

    def prob(self, x: V, **kwargs) -> V:
        return jnp.exp(self.log_prob(x, **kwargs))  # ty:ignore[invalid-return-type]


class Normal(BasePDF):
    def __init__(self, mean: V, width: V):
        self.mean = float_array(mean)
        self.width = float_array(width)

    def log_prob(self, x: V) -> V:
        logpdf_max = jax.scipy.stats.norm.logpdf(
            self.mean, loc=self.mean, scale=self.width
        )
        unnormalized = jax.scipy.stats.norm.logpdf(x, loc=self.mean, scale=self.width)
        return unnormalized - logpdf_max  # ty:ignore[invalid-return-type]

    def cdf(self, x: V) -> V:
        return jax.scipy.stats.norm.cdf(x, loc=self.mean, scale=self.width)  # ty:ignore[invalid-return-type]

    def inv_cdf(self, x: V) -> V:
        return jax.scipy.stats.norm.ppf(x, loc=self.mean, scale=self.width)  # ty:ignore[invalid-return-type]

    def __evermore_from_unit_normal__(self, x: V) -> V:
        return self.mean + self.width * x  # ty:ignore[invalid-return-type]

    def sample(self, key: PRNGKeyArray, shape: Shape) -> Float[Array, ...]:
        # sample parameter from pdf
        return self.__evermore_from_unit_normal__(jax.random.normal(key, shape=shape))


class PoissonBase(BasePDF):
    def __init__(self, lamb: V):
        self.lamb = float_array(lamb)


class PoissonDiscrete(PoissonBase):
    """Poisson distribution with discrete support.

    Float inputs are floored to the nearest integer, matching the behaviour of
    libraries such as SciPy or RooFit.
    """

    def log_prob(
        self,
        x: V,
        normalize: bool = True,
    ) -> V:
        k = jnp.floor(x)

        # plain evaluation of the pmf
        unnormalized = jax.scipy.stats.poisson.logpmf(k, self.lamb)
        if not normalize:
            return unnormalized  # ty:ignore[invalid-return-type]

        # when normalizing, divide (subtract in log space) by maximum over k range
        logpdf_max = jax.scipy.stats.poisson.logpmf(k, k)
        return unnormalized - logpdf_max  # ty:ignore[invalid-return-type]

    def cdf(self, x: V) -> V:
        # no need to round x to k, already done by cdf library function
        return jax.scipy.stats.poisson.cdf(x, self.lamb)  # ty:ignore[invalid-return-type]

    def inv_cdf(self, x: V, rounding: DiscreteRounding = "floor") -> V:
        # define starting point for search from normal approximation
        def start_fn(x: V) -> V:
            return jnp.floor(
                self.lamb + jax.scipy.stats.norm.ppf(x) * jnp.sqrt(self.lamb)
            )  # ty:ignore[invalid-return-type]

        # define the cdf function
        def cdf_fn(k: V) -> V:
            return jax.scipy.stats.poisson.cdf(k, self.lamb)  # ty:ignore[invalid-return-type]

        return discrete_inv_cdf_search(
            x,
            cdf_fn=cdf_fn,
            start_fn=start_fn,
            rounding=rounding,
        )

    def sample(self, key: PRNGKeyArray, shape: Shape) -> Float[Array, ...]:
        return jax.random.poisson(key, self.lamb, shape=shape)


class PoissonContinuous(PoissonBase):
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

        # plain evaluation of the pdf
        unnormalized = _log_prob(x, lamb)
        if not normalize:
            return unnormalized

        # when normalizing, divide (subtract in log space) by maximum over a range
        # that depends on whether the mode is shifted
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


# alias for rounding literals
DiscreteRounding = Literal["floor", "ceil", "closest"]
known_roundings = frozenset(DiscreteRounding.__args__)  # type: ignore[attr-defined]


def discrete_inv_cdf_search(
    x: V,
    cdf_fn: Callable[[V], V],
    start_fn: Callable[[V], V],
    rounding: DiscreteRounding,
) -> V:
    """Computes an inverse CDF for discrete distributions via iterative search.

    Args:
        x: Values between 0 and 1 for which the inverse CDF should be evaluated.
        cdf_fn: Callable returning the cumulative distribution evaluated at a given integer.
        start_fn: Callable providing an initial guess for the search (usually via an approximation).
        rounding: Strategy used when the target value lies between two integers.

    Returns:
        V: Integral values with the same shape as ``x`` that correspond to the requested quantiles.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.scipy.stats
        >>> lamb = 5.0
        >>> def start_fn(q):
        ...     return jnp.floor(lamb + jax.scipy.stats.norm.ppf(q) * jnp.sqrt(lamb))
        >>> def cdf_fn(k):
        ...     return jax.scipy.stats.poisson.cdf(k, lamb)
        >>> discrete_inv_cdf_search(jnp.array(0.9), cdf_fn, start_fn, \"floor\")
        Array(7., dtype=float32)
    """
    # store masks for injecting exact values for known edge cases later on
    # inject 0 for x == 0
    zero_mask = x == 0.0
    # inject inf for x == 1
    inf_mask = x == 1.0
    # inject nan for ~(0 < x < 1) or non-finite values
    nan_mask = (x < 0.0) | (x > 1.0) | ~jnp.isfinite(x)

    # setup stopping condition and iteration body for the iterative search
    # note: functions are defined for scalar values and then vmap'd, with results being reshaped
    def cond_fn(val):
        *_, stop = val
        return ~jnp.any(stop)

    def body_fn(val):
        k, target_itg, prev_itg, stop = val
        # compute the current integral
        itg = cdf_fn(k)
        # special case: itg is the exact solution
        stop |= itg == target_itg
        # if no previous integral is available or if we have not yet "cornered" the target value
        # with the current and previous integrals, make a step in the right direction
        make_step = (
            (prev_itg < 0)
            | ((prev_itg < itg) & (itg < target_itg))
            | ((target_itg < itg) & (itg < prev_itg))
        )
        step = jnp.where(~stop & make_step, jnp.sign(target_itg - itg), 0)
        k += step
        # if target_itg is between the computed integrals we can now find the correct k
        # note: k might be subject to a shift by +1 or -1, depending on the stride and rounding
        k_found = ~stop & ~make_step

        # we're using python >=3.11 :)
        match rounding:
            case "floor":
                k_shift = jnp.where(k_found & (itg > target_itg), -1, 0)
            case "ceil":
                k_shift = jnp.where(k_found & (prev_itg > target_itg), 1, 0)
            case "closest":
                k_shift = jnp.where(
                    k_found & (abs(itg - target_itg) > abs(prev_itg - target_itg)),
                    jnp.sign(prev_itg - itg),
                    0,
                )
            case _:
                msg = f"unknown rounding '{rounding}' mode, expected one of {', '.join(known_roundings)}"  # type: ignore[unreachable]
                raise ValueError(msg)

        k += k_shift
        # update the stop flag and end
        stop |= k_found
        return (k, target_itg, itg, stop)

    def search(start_k, target_itg, stop):
        prev_itg = -jnp.ones_like(target_itg)
        val = (start_k, target_itg, prev_itg, stop)
        return jax.lax.while_loop(cond_fn, body_fn, val)[0]

    # jnp.vectorize is auto-vmapping over all axes of its arguments,
    # see: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.vectorize.html#jax.numpy.vectorize
    vsearch = jnp.vectorize(search)

    # define starting point and stop flag (eagerly skipping edge cases), then search
    start_k = start_fn(x)
    stop = zero_mask | inf_mask | nan_mask
    k = vsearch(start_k, x, stop)

    # inject known values for edge cases
    k = jnp.where(zero_mask, 0.0, k)
    k = jnp.where(inf_mask, jnp.inf, k)
    k = jnp.where(nan_mask, jnp.nan, k)

    return k  # noqa: RET504
