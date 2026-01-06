from __future__ import annotations

import abc
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import checkify
from jaxtyping import PyTree

from evermore.parameters.filter import is_parameter
from evermore.parameters.parameter import BaseParameter, V
from evermore.util import float_array

__all__ = [
    "BaseParameterTransformation",
    "MinuitTransform",
    "SoftPlusTransform",
    "unwrap",
    "wrap",
]


def __dir__():
    return __all__


# error handling taken from https://github.com/JaxGaussianProcesses/GPJax/blob/main/gpjax/parameters.py
def _safe_assert(fn: tp.Callable, value: V, **kwargs) -> None:
    error, _ = fn(value, **kwargs)
    checkify.check_error(error)
    return


@checkify.checkify
def _check_in_bounds(value: V, lower: V, upper: V) -> None:
    """Check if a value is bounded between lower and upper.

    Args:
        value: The value to check.
        lower: The lower bound.
        upper: The upper bound.

    Raises:
        ValueError: If any element of value is outside the bounds.
    """
    checkify.check(
        jnp.all((value > lower) & (value < upper)),
        "value needs to be bounded between {lower} and {upper}, got {value}",
        value=value,
        lower=lower,
        upper=upper,
    )


@checkify.checkify
def _check_is_finite(value: V) -> None:
    """Check if a value is finite.

    Args:
        value: The value to check.

    Raises:
        ValueError: If any element of value is not finite.
    """
    checkify.check(
        jnp.all(jnp.isfinite(value)),
        "value needs to be finite",
    )


@checkify.checkify
def _check_is_non_negative(value: V) -> None:
    """Check if a value is element-wise non-negative.

    Args:
        value: Values to validate.

    Raises:
        ValueError: If any element is negative.
    """
    checkify.check(
        jnp.all(value >= 0), "value needs to be non-negative, got {value}", value=value
    )


def unwrap(params: PyTree[BaseParameter]) -> PyTree[BaseParameter]:
    """Applies registered transformations to move parameters into unconstrained space.

    Args:
        params: PyTree that may contain parameters with attached transformations.

    Returns:
        PyTree where each parameter has been transformed via ``unwrap``.
    """

    def _unwrap(path, param: BaseParameter[V]) -> BaseParameter[V]:
        del path  # unused
        if param.transform is None:
            return param
        return param.transform.unwrap(param)

    graphdef, params_state, rest = nnx.split(params, is_parameter, ...)
    params_state_t = nnx.map_state(_unwrap, params_state)
    return nnx.merge(graphdef, params_state_t, rest)


def wrap(params: PyTree[BaseParameter]) -> PyTree[BaseParameter]:
    """Applies registered transformations to move parameters back to constrained space.

    Args:
        params: PyTree that may contain parameters with attached transformations.

    Returns:
        PyTree where each parameter has been transformed via ``wrap``.
    """

    def _wrap(path, param: BaseParameter[V]) -> BaseParameter[V]:
        del path  # unused
        if param.transform is None:
            return param
        return param.transform.wrap(param)

    graphdef, params_state, rest = nnx.split(params, is_parameter, ...)
    params_state_t = nnx.map_state(_wrap, params_state)
    return nnx.merge(graphdef, params_state_t, rest)


class BaseParameterTransformation(nnx.Module):
    """Abstract interface for parameter transformations.

    Subclasses provide ``unwrap``/``wrap`` implementations that translate between
    constrained and unconstrained representations of parameters.
    """

    @abc.abstractmethod
    def unwrap(self, parameter: BaseParameter[V]) -> BaseParameter[V]:
        """Transforms a parameter from constrained to unconstrained space.

        Args:
            parameter: Parameter to transform.

        Returns:
            BaseParameter: Transformed parameter instance.
        """

    @abc.abstractmethod
    def wrap(self, parameter: BaseParameter[V]) -> BaseParameter[V]:
        """Transforms a parameter from unconstrained space back to its original domain.

        Args:
            parameter: Parameter to transform.

        Returns:
            BaseParameter: Parameter in its original space.
        """


class MinuitTransform(BaseParameterTransformation):
    """Implements MINUIT-style transformations for bounded parameters.

    Both lower and upper bounds must be finite for the transformation to be well-defined.

    References:
        MINUIT User's Guide, Section 1.2.1 ``The transformation for parameters with limits``.

    Examples:
        >>> import evermore as evm
        >>> from evermore.parameters import transform as tr
        >>> minuit = tr.MinuitTransform()
        >>> params = {
        ...     "a": evm.Parameter(2.0, lower=-0.1, upper=2.2, transform=minuit),
        ...     "b": evm.Parameter(0.1, lower=0.0, upper=1.1, transform=minuit),
        ... }
        >>> unconstrained = tr.unwrap(params)
        >>> restored = tr.wrap(unconstrained)
        >>> restored["a"].value == params["a"].value
        Array(True, dtype=bool)
    """

    def _check_and_regularize(self, parameter: BaseParameter[V]) -> tuple[V, V, V]:
        # this is not allowed here
        if (parameter.lower is None and parameter.upper is not None) or (
            parameter.lower is not None and parameter.upper is None
        ):
            msg = f"{parameter} must have both lower and upper boundaries set, or none of them."
            raise ValueError(msg)

        value = float_array(parameter.value)
        lower = float_array(parameter.lower)
        upper = float_array(parameter.upper)

        # check for finite boundaries
        _safe_assert(_check_is_finite, lower)
        _safe_assert(_check_is_finite, upper)

        return value, lower, upper  # ty:ignore[invalid-return-type]

    def unwrap(self, parameter: BaseParameter[V]) -> BaseParameter[V]:
        # short-cut
        if parameter.lower is None and parameter.upper is None:
            return parameter

        value, lower, upper = self._check_and_regularize(parameter)

        # for unwrapping, we need to make sure the value is within the boundaries initially
        _safe_assert(
            _check_in_bounds,
            value,
            lower=lower,
            upper=upper,
        )

        # this formula turns user-provided "external" parameter values into "internal" values
        new_value = jnp.arcsin(
            2.0 * (value - lower) / (upper - lower)  # type: ignore[operator]
            - 1.0
        )
        return parameter.replace(value=new_value)  # ty:ignore[invalid-return-type]

    def wrap(self, parameter: BaseParameter[V]) -> BaseParameter[V]:
        # short-cut
        if parameter.lower is None and parameter.upper is None:
            return parameter

        value, lower, upper = self._check_and_regularize(parameter)

        # this formula turns "internal" parameter values into "external" values
        new_value = lower + (upper - lower) / 2.0 * (jnp.sin(value) + 1.0)
        return parameter.replace(value=new_value)  # ty:ignore[invalid-return-type]


class SoftPlusTransform(BaseParameterTransformation):
    """Ensures parameters remain positive by using the softplus bijection.

    This transformation does not require explicit bounds; ``unwrap`` maps to the
    unconstrained real line and ``wrap`` maps back to the positive reals.

    Examples:
        >>> import evermore as evm
        >>> from evermore.parameters import transform as tr
        >>> positive = tr.SoftPlusTransform()
        >>> params = {
        ...     "a": evm.Parameter(2.0, transform=positive),
        ...     "b": evm.Parameter(0.1, transform=positive),
        ... }
        >>> unconstrained = tr.unwrap(params)
        >>> restored = tr.wrap(unconstrained)
        >>> restored["b"].value
        Array(0.1, dtype=float32)
    """

    def unwrap(self, parameter: BaseParameter[V]) -> BaseParameter[V]:
        # from: https://github.com/danielward27/paramax/blob/main/paramax/utils.py
        """Applies the inverse softplus transformation after validating the value."""
        value = float_array(parameter.value)

        _safe_assert(_check_is_non_negative, value)

        new_value = jnp.log(-jnp.expm1(-value)) + value
        return parameter.replace(value=new_value)  # ty:ignore[invalid-return-type]

    def wrap(self, parameter: BaseParameter[V]) -> BaseParameter[V]:
        new_value = jax.nn.softplus(float_array(parameter.value))
        return parameter.replace(value=new_value)  # ty:ignore[invalid-return-type]
