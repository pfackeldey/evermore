from __future__ import annotations

import abc
import typing as tp

import equinox as eqx
import jax
import jax.numpy as jnp
from jax._src.random import Shape
from jax.scipy.special import digamma, gammaln, xlogy
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from evermore.util import atleast_1d_float_array
from evermore.visualization import SupportsTreescope

__all__ = [
    "PDF",
    "Normal",
    "PoissonBase",
    "PoissonContinuous",
    "PoissonDiscrete",
]


# alias for rounding literals
DiscreteRounding = tp.Literal["floor", "ceil", "closest"]


def __dir__():
    return __all__


@tp.runtime_checkable
class ImplementsFromUnitNormalConversion(tp.Protocol):
    def __evermore_from_unit_normal__(self, x: Array) -> Array: ...


class PDF(eqx.Module, SupportsTreescope):
    @abc.abstractmethod
    def log_prob(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def cdf(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def inv_cdf(self, x: Array) -> Array: ...

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

    def cdf(self, x: Array) -> Array:
        return jax.scipy.stats.norm.cdf(x, loc=self.mean, scale=self.width)

    def inv_cdf(self, x: Array) -> Array:
        return jax.scipy.stats.norm.ppf(x, loc=self.mean, scale=self.width)

    def __evermore_from_unit_normal__(self, x: Array) -> Array:
        return self.mean + self.width * x

    def sample(self, key: PRNGKeyArray, shape: Shape | None = None) -> Array:
        # jax.random.normal does not accept None shape
        if shape is None:
            shape = ()
        # sample parameter from pdf
        return self.__evermore_from_unit_normal__(jax.random.normal(key, shape=shape))


class PoissonBase(PDF):
    lamb: Array = eqx.field(converter=atleast_1d_float_array)


class PoissonDiscrete(PoissonBase):
    """
    Poisson distribution with discrete support. Float inputs are floored to the nearest integer,
    similar to the behavior implemented in other libraries like SciPy or RooFit.
    """

    def log_prob(self, x: Array, normalize: bool = True) -> Array:
        # explicit rounding
        k = jnp.floor(x)

        # plain evaluation of the pmf
        unnormalized = jax.scipy.stats.poisson.logpmf(k, self.lamb)
        if not normalize:
            return unnormalized

        # when normalizing, divide (subtract in log space) by maximum over k range
        logpdf_max = jax.scipy.stats.poisson.logpmf(k, k)
        return unnormalized - logpdf_max

    def cdf(self, x: Array) -> Array:
        # no need to round x to k, already done by cdf library function
        return jax.scipy.stats.poisson.cdf(x, self.lamb)

    def inv_cdf(self, x: Array, rounding: DiscreteRounding = "floor") -> Array:
        # define starting point for search from normal approximation
        def start_fn(x):
            return jnp.floor(
                self.lamb + jax.scipy.stats.norm.ppf(x) * jnp.sqrt(self.lamb)
            )

        # define the cdf function
        def cdf_fn(k):
            return jax.scipy.stats.poisson.cdf(k, self.lamb)

        return discrete_inv_cdf_search(
            x,
            cdf_fn=cdf_fn,
            start_fn=start_fn,
            rounding=rounding,
        )

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

        # plain evaluation of the pdf
        unnormalized = _log_prob(x, lamb)
        if not normalize:
            return unnormalized

        # when normalizing, divide (subtract in log space) by maximum over a range
        # that depends on whether the mode is shifted
        args = (self.lamb, lamb) if shift_mode else (x, x)
        logpdf_max = _log_prob(*args)
        return unnormalized - logpdf_max

    def cdf(self, x: Array) -> Array:
        err = f"{self.__class__.__name__} does not support cdf"
        raise Exception(err)

    def inv_cdf(self, x: Array) -> Array:
        err = f"{self.__class__.__name__} does not support inv_cdf"
        raise Exception(err)

    def sample(self, key: PRNGKeyArray, shape: Shape | None = None) -> Array:
        msg = f"{self.__class__.__name__} does not support sampling, use PoissonDiscrete instead"
        raise Exception(msg)


def discrete_inv_cdf_search(
    x: Array,
    cdf_fn: tp.Callable[[ArrayLike], ArrayLike],
    start_fn: tp.Callable[[ArrayLike], ArrayLike],
    rounding: DiscreteRounding,
) -> Array:
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
        x (Array): Integral values to compute the inverse CDF for.
        cdf_fn (Callable): A callable representing the discrete CDF function. It is called with a
            single argument and supposed to return the CDF value for that argument.
        start_fn (Callable): A callable that provides a starting point for the search. It is called
            with a reshaped representation of *x*.
        rounding (DiscreteRounding): One of "floor", "ceil" or "closest".

    Returns:
        Array: The computed inverse CDF values in the same shape as *x*.
    """
    # flatten input
    x_shape = x.shape
    x = jnp.reshape(x, (-1, 1))

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
        stop = val[-1]
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
        if rounding == "floor":
            k_shift = jnp.where(k_found & (itg > target_itg), -1, 0)
        elif rounding == "ceil":
            k_shift = jnp.where(k_found & (prev_itg > target_itg), 1, 0)
        else:  # "closest"
            k_shift = jnp.where(
                k_found & (abs(itg - target_itg) > abs(prev_itg - target_itg)),
                jnp.sign(prev_itg - itg),
                0,
            )
        k += k_shift
        # update the stop flag and end
        stop |= k_found
        return (k, target_itg, itg, stop)

    def search(start_k, target_itg, stop):
        prev_itg = -jnp.ones_like(target_itg)
        val = (start_k, target_itg, prev_itg, stop)
        return jax.lax.while_loop(cond_fn, body_fn, val)[0]

    # vmap
    vsearch = jax.vmap(search, in_axes=(0, 0, 0))

    # define starting point and stop flag (eagerly skipping edge cases), then search
    start_k = start_fn(x)
    stop = zero_mask | inf_mask | nan_mask
    k = vsearch(start_k, x, stop)

    # inject known values for edge cases
    k = jnp.where(zero_mask, 0.0, k)
    k = jnp.where(inf_mask, jnp.inf, k)
    k = jnp.where(nan_mask, jnp.nan, k)

    # reshape to input shape
    return jnp.reshape(k, x_shape)
