from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from evermore.parameters.filter import is_parameter
from evermore.parameters.parameter import (
    AbstractParameter,
    V,
    replace_value,
)
from evermore.parameters.tree import PT, only

__all__ = [
    "AbstractParameterTransformation",
    "MinuitTransform",
    "SoftPlusTransform",
    "unwrap",
    "wrap",
]


def __dir__():
    return __all__


def unwrap(params: PT) -> PT:
    """
    Unwraps the parameters in the given PyTree by applying their respective transformations.

    This function traverses the PyTree of parameters and applies the `unwrap` method of each parameter's
    transformation, if it exists. If a parameter does not have a transformation, it remains unchanged.

    Args:
        params (PT): A PyTree of parameters to be unwrapped.

    Returns:
        PT: A new PyTree with the parameters unwrapped.
    """

    def _unwrap(param: AbstractParameter[V]) -> AbstractParameter[V]:
        if param.transform is None:
            return param
        return param.transform.unwrap(param)

    return jax.tree.map(_unwrap, only(params, is_parameter), is_leaf=is_parameter)


def wrap(params: PT) -> PT:
    """
    Wraps the parameters in the given PyTree by applying their respective transformations.
    This is the inverse operation of `unwrap`.

    This function traverses the PyTree of parameters and applies the `wrap` method of each parameter's
    transformation, if it exists. If a parameter does not have a transformation, it remains unchanged.

    Args:
        params (PT): A PyTree of parameters to be wrapped.

    Returns:
        PT: A new PyTree with the parameters wrapped.
    """

    def _wrap(param: AbstractParameter[V]) -> AbstractParameter[V]:
        if param.transform is None:
            return param
        return param.transform.wrap(param)

    return jax.tree.map(_wrap, only(params, is_parameter), is_leaf=is_parameter)


class AbstractParameterTransformation(eqx.Module):
    """
    Abstract base class for parameter transformations.

    This class defines the interface for parameter transformations, which are used to map parameters
    between different spaces (e.g., from constrained to unconstrained space). Subclasses must implement
    the `unwrap` and `wrap` methods to define the specific transformation logic.
    """

    @abc.abstractmethod
    def unwrap(self, parameter: AbstractParameter[V]) -> AbstractParameter[V]:
        """
        Transform a parameter from its meaningful (e.g. bounded) space to the real unconstrained space.

        Args:
            parameter (AbstractParameter): The parameter to be transformed.

        Returns:
            AbstractParameter: The transformed parameter.
        """

    @abc.abstractmethod
    def wrap(self, parameter: AbstractParameter[V]) -> AbstractParameter[V]:
        """
        Transform a parameter from the real unconstrained space back to its meaningful (e.g. bounded) space. (Inverse of `unwrap`)

        Args:
            parameter (AbstractParameter): The parameter to be transformed.

        Returns:
            AbstractParameter: The parameter transformed back to its original space.
        """


class MinuitTransform(AbstractParameterTransformation):
    """
    Transform parameters based on Minuit's conventions. This transformation is used to map parameters with finite
    lower and upper boundaries to an unconstrained space. Both lower and upper boundaries
    are required and must be finite.

    Use `unwrap` to transform parameters into the unconstrained space and `wrap` to transform them back into the bounded space.

    Reference:
    https://root.cern.ch/download/minuit.pdf (Sec. 1.2.1 The transformation for parameters with limits.)

    Example:

        .. code-block:: python

            import evermore as evm
            import wadler_lindig as wl

            from evermore.parameters.transform import MinuitTransform, unwrap, wrap

            minuit_transform = MinuitTransform()
            pytree = {
                "a": evm.Parameter(2.0, lower=-0.1, upper=2.2, transform=minuit_transform),
                "b": evm.Parameter(0.1, lower=0.0, upper=1.1, transform=minuit_transform),
            }

            # unwrap (or "transform")
            pytree_t = unwrap(pytree)
            # wrap back (or "inverse transform")
            pytree_tt = wrap(pytree_t)

            wl.pprint(pytree, width=250, short_arrays=False)
            # {
            #   'a': Parameter(raw_value=ValueAttr(value=Array(2., dtype=float32)), name=None, lower=Array(-0.1, dtype=float32), upper=Array(2.2, dtype=float32), prior=None, frozen=False, transform=MinuitTransform(), tags=frozenset()),
            #   'b': Parameter(raw_value=ValueAttr(value=Array(0.1, dtype=float32)), name=None, lower=Array(0., dtype=float32), upper=Array(1.1, dtype=float32), prior=None, frozen=False, transform=MinuitTransform(), tags=frozenset())
            # }

            wl.pprint(pytree_t, width=250, short_arrays=False)
            # {
            #   'a': Parameter(raw_value=ValueAttr(value=Array(0.9721281, dtype=float32)), name=None, lower=Array(-0.1, dtype=float32), upper=Array(2.2, dtype=float32), prior=None, frozen=False, transform=MinuitTransform(), tags=frozenset()),
            #   'b': Parameter(raw_value=ValueAttr(value=Array(-0.95824164, dtype=float32)), name=None, lower=Array(0., dtype=float32), upper=Array(1.1, dtype=float32), prior=None, frozen=False, transform=MinuitTransform(), tags=frozenset())
            # }

            wl.pprint(pytree_tt, width=250, short_arrays=False)
            # {
            #   'a': Parameter(raw_value=ValueAttr(value=Array(1.9999999, dtype=float32)), name=None, lower=Array(-0.1, dtype=float32), upper=Array(2.2, dtype=float32), prior=None, frozen=False, transform=MinuitTransform(), tags=frozenset()),
            #   'b': Parameter(raw_value=ValueAttr(value=Array(0.09999997, dtype=float32)), name=None, lower=Array(0., dtype=float32), upper=Array(1.1, dtype=float32), prior=None, frozen=False, transform=MinuitTransform(), tags=frozenset())
            # }
    """

    def _check(self, parameter: AbstractParameter[V]) -> AbstractParameter[V]:
        if (parameter.lower is None and parameter.upper is not None) or (
            parameter.lower is not None and parameter.upper is None
        ):
            msg = f"{parameter} must have both lower and upper boundaries set, or none of them."
            raise ValueError(msg)
        lower = parameter.lower
        upper = parameter.upper
        # check for finite boundaries
        error_msg = f"Bounds of {parameter} must be finite, got {parameter.lower=}, {parameter.upper=}."
        parameter = eqx.error_if(
            parameter,
            ~jnp.isfinite(lower),
            error_msg,
        )
        return eqx.error_if(
            parameter,
            ~jnp.isfinite(upper),
            error_msg,
        )

    def unwrap(self, parameter: AbstractParameter[V]) -> AbstractParameter[V]:
        # short-cut
        if parameter.lower is None and parameter.upper is None:
            return parameter

        # for unwrapping, we need to make sure the value is within the boundaries initially
        error_msg = f"The value of {parameter} is exactly at or outside the boundaries [{parameter.lower}, {parameter.upper}]."
        parameter = eqx.error_if(
            parameter,
            parameter.value <= parameter.lower,
            error_msg,
        )
        parameter = eqx.error_if(
            parameter,
            parameter.value >= parameter.upper,
            error_msg,
        )

        parameter = self._check(parameter)
        # this formula turns user-provided "external" parameter values into "internal" values
        value_t = jnp.arcsin(
            2.0
            * (parameter.value - parameter.lower)
            / (parameter.upper - parameter.lower)
            - 1.0
        )
        return replace_value(parameter, value_t)

    def wrap(self, parameter: AbstractParameter[V]) -> AbstractParameter[V]:
        # short-cut
        if parameter.lower is None and parameter.upper is None:
            return parameter

        parameter = self._check(parameter)
        # this formula turns "internal" parameter values into "external" values
        value_t = parameter.lower + (parameter.upper - parameter.lower) / 2 * (
            jnp.sin(parameter.value) + 1
        )
        return replace_value(parameter, value_t)


class SoftPlusTransform(AbstractParameterTransformation):
    """
    Applies the softplus transformation to parameters, projecting them from real space (R) to positive space (R+).
    This transformation is useful for enforcing the positivity of parameters and does not require lower or upper boundaries.

    Use `unwrap` to transform parameters into the unconstrained real space and `wrap` to transform them back into the positive real space.

    Example:

    .. code-block:: python

        import evermore as evm
        import wadler_lindig as wl

        from evermore.parameters.transform import SoftPlusTransform, unwrap, wrap

        enforce_positivity = SoftPlusTransform()
        pytree = {
            "a": evm.Parameter(2.0, transform=enforce_positivity),
            "b": evm.Parameter(0.1, transform=enforce_positivity),
        }

        # unwrap (or "transform")
        pytree_t = unwrap(pytree)
        # wrap back (or "inverse transform")
        pytree_tt = wrap(pytree_t)

        wl.pprint(pytree, width=250, short_arrays=False)
        # {
        #   'a': Parameter(raw_value=ValueAttr(value=Array(2., dtype=float32)), name=None, lower=None, upper=None, prior=None, frozen=False, transform=SoftPlusTransform(), tags=frozenset()),
        #   'b': Parameter(raw_value=ValueAttr(value=Array(0.1, dtype=float32)), name=None, lower=None, upper=None, prior=None, frozen=False, transform=SoftPlusTransform(), tags=frozenset())
        # }

        wl.pprint(pytree_t, width=250, short_arrays=False)
        # {
        #   'a': Parameter(raw_value=ValueAttr(value=Array(1.8545866, dtype=float32)), name=None, lower=None, upper=None, prior=None, frozen=False, transform=SoftPlusTransform(), tags=frozenset()),
        #   'b': Parameter(raw_value=ValueAttr(value=Array(-2.2521687, dtype=float32)), name=None, lower=None, upper=None, prior=None, frozen=False, transform=SoftPlusTransform(), tags=frozenset())
        # }

        wl.pprint(pytree_tt, width=250, short_arrays=False)
        # {
        #   'a': Parameter(raw_value=ValueAttr(value=Array(2., dtype=float32)), name=None, lower=None, upper=None, prior=None, frozen=False, transform=SoftPlusTransform(), tags=frozenset()),
        #   'b': Parameter(raw_value=ValueAttr(value=Array(0.09999998, dtype=float32)), name=None, lower=None, upper=None, prior=None, frozen=False, transform=SoftPlusTransform(), tags=frozenset())
        # }
    """

    def unwrap(self, parameter: AbstractParameter[V]) -> AbstractParameter[V]:
        # from: https://github.com/danielward27/paramax/blob/main/paramax/utils.py
        """The inverse of the softplus function, checking for positive inputs."""
        parameter = eqx.error_if(
            parameter,
            parameter.value < 0,
            "Expected positive inputs to inv_softplus.",
        )
        value_t = jnp.log(-jnp.expm1(-parameter.value)) + parameter.value
        return replace_value(parameter, value_t)

    def wrap(self, parameter: AbstractParameter[V]) -> AbstractParameter[V]:
        value_t = jax.nn.softplus(parameter.value)
        return replace_value(parameter, value_t)
