from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Generic, Literal, Protocol, runtime_checkable

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
    Poisson distribution with discrete support. Float inputs are floored to the nearest integer,
    similar to the behavior implemented in other libraries like SciPy or RooFit.
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
            return unnormalized

        # when normalizing, divide (subtract in log space) by maximum over k range
        logpdf_max = jax.scipy.stats.poisson.logpmf(k, k)
        return unnormalized - logpdf_max

    def cdf(self, x: V) -> V:
        # no need to round x to k, already done by cdf library function
        return jax.scipy.stats.poisson.cdf(x, self.lamb)

    def inv_cdf(self, x: V, rounding: DiscreteRounding = "floor") -> V:
        # define starting point for search from normal approximation
        def start_fn(x: V) -> V:
            return jnp.floor(
                self.lamb + jax.scipy.stats.norm.ppf(x) * jnp.sqrt(self.lamb)
            )

        # define the cdf function
        def cdf_fn(k: V) -> V:
            return jax.scipy.stats.poisson.cdf(k, self.lamb)

        return discrete_inv_cdf_search(
            x,
            cdf_fn=cdf_fn,
            start_fn=start_fn,
            rounding=rounding,
        )

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
    """
    Computes the inverse CDF (percent point function) at integral values *x* for a discrete CDF
    distribution *cdf* using an iterative search strategy. The search starts at values provided by
    *start_fn* and progresses in integer steps towards the target values.

    .. code-block:: python

        # this example mimics the PoissonDiscrete.inv_cdf implementation

        import jax
        import jax.numpy as jnp
        import evermore as evm

        # parameter of the poisson distribution
        lamb = 5.0

        # the normal approximation is a good starting point
        def start_fn(x):
            return jnp.floor(lamb + jax.scipy.stats.norm.ppf(x) * jnp.sqrt(lamb))


        # define the cdf function
        def cdf_fn(k):
            return jax.scipy.stats.poisson.cdf(k, lamb)


        k = discrete_inv_cdf_search(jnp.array(0.9), cdf_fn, start_fn, "floor")
        # -> 7.0

    Args:
        x (V): Integral values to compute the inverse CDF for.
        cdf_fn (Callable): A callable representing the discrete CDF function. It is called with a
            single argument and supposed to return the CDF value for that argument.
        start_fn (Callable): A callable that provides a starting point for the search. It is called
            with a reshaped representation of *x*.
        rounding (DiscreteRounding): One of "floor", "ceil" or "closest".

    Returns:
        V: The computed inverse CDF values in the same shape as *x*.
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
