from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from evermore.parameters.parameter import Parameter, _params_map, _ParamsTree
from evermore.util import _missing

__all__ = [
    "MinuitTransform",
    "ParameterTransformation",
    "SoftPlusTransform",
    "unwrap",
    "wrap",
]


def __dir__():
    return __all__


def unwrap(params: _ParamsTree) -> _ParamsTree:
    """
    Unwraps the parameters in the given PyTree by applying their respective transformations.

    This function traverses the PyTree of parameters and applies the `unwrap` method of each parameter's
    transformation, if it exists. If a parameter does not have a transformation, it remains unchanged.

    Args:
        params (_ParamsTree): A PyTree of parameters to be unwrapped.

    Returns:
        _ParamsTree: A new PyTree with the parameters unwrapped.
    """

    def _unwrap(param: Parameter) -> Parameter:
        if param.transform is None:
            return param
        return param.transform.unwrap(param)

    return _params_map(_unwrap, params)


def wrap(params: _ParamsTree) -> _ParamsTree:
    """
    Wraps the parameters in the given PyTree by applying their respective transformations.
    This is the inverse operation of `unwrap`.

    This function traverses the PyTree of parameters and applies the `wrap` method of each parameter's
    transformation, if it exists. If a parameter does not have a transformation, it remains unchanged.

    Args:
        params (_ParamsTree): A PyTree of parameters to be wrapped.

    Returns:
        _ParamsTree: A new PyTree with the parameters wrapped.
    """

    def _wrap(param: Parameter) -> Parameter:
        if param.transform is None:
            return param
        return param.transform.wrap(param)

    return _params_map(_wrap, params)


class ParameterTransformation(eqx.Module):
    """
    Abstract base class for parameter transformations.

    This class defines the interface for parameter transformations, which are used to map parameters
    between different spaces (e.g., from constrained to unconstrained space). Subclasses must implement
    the `unwrap` and `wrap` methods to define the specific transformation logic.
    """

    @abc.abstractmethod
    def unwrap(self, parameter: Parameter) -> Parameter:
        """
        Transform a parameter from its meaningful (e.g. bounded) space to the real unconstrained space.

        Args:
            parameter (Parameter): The parameter to be transformed.

        Returns:
            Parameter: The transformed parameter.
        """

    @abc.abstractmethod
    def wrap(self, parameter: Parameter) -> Parameter:
        """
        Transform a parameter from the real unconstrained space back to its meaningful (e.g. bounded) space. (Inverse of `unwrap`)

        Args:
            parameter (Parameter): The parameter to be transformed.

        Returns:
            Parameter: The parameter transformed back to its original space.
        """


class MinuitTransform(ParameterTransformation):
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

            wl.pprint(pytree, width=150, short_arrays=False)
            # {
            #   'a': Parameter(value=Array([2.], dtype=float32), lower=-0.1, upper=2.2, transform=MinuitTransform()),
            #   'b': Parameter(value=Array([0.1], dtype=float32), lower=0.0, upper=1.1, transform=MinuitTransform())
            # }

            wl.pprint(pytree_t, width=150, short_arrays=False)
            # {
            #   'a': Parameter(value=Array([0.9721281], dtype=float32), lower=-0.1, upper=2.2, transform=MinuitTransform()),
            #   'b': Parameter(value=Array([-0.95824164], dtype=float32), lower=0.0, upper=1.1, transform=MinuitTransform())
            # }

            wl.pprint(pytree_tt, width=150, short_arrays=False)
            # {
            #   'a': Parameter(value=Array([1.9999999], dtype=float32), lower=-0.1, upper=2.2, transform=MinuitTransform()),
            #   'b': Parameter(value=Array([0.09999997], dtype=float32), lower=0.0, upper=1.1, transform=MinuitTransform())
            # }
    """

    def _check(self, parameter: Parameter) -> Parameter:
        if (parameter.lower is None and parameter.upper is not None) or (
            parameter.lower is not None and parameter.upper is None
        ):
            msg = f"{parameter} must have both lower and upper boundaries set, or none of them."
            raise ValueError(msg)
        lower: ArrayLike = parameter.lower  # type: ignore[assignment]
        upper: ArrayLike = parameter.upper  # type: ignore[assignment]
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

    def unwrap(self, parameter: Parameter) -> Parameter:
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
            * (parameter.value - parameter.lower)  # type: ignore[operator]
            / (parameter.upper - parameter.lower)  # type: ignore[operator]
            - 1.0
        )
        return eqx.tree_at(
            lambda p: p.value, parameter, value_t, is_leaf=lambda leaf: leaf is _missing
        )

    def wrap(self, parameter: Parameter) -> Parameter:
        # short-cut
        if parameter.lower is None and parameter.upper is None:
            return parameter

        parameter = self._check(parameter)
        # this formula turns "internal" parameter values into "external" values
        value_t = parameter.lower + (parameter.upper - parameter.lower) / 2 * (  # type: ignore[operator]
            jnp.sin(parameter.value) + 1
        )
        return eqx.tree_at(
            lambda p: p.value, parameter, value_t, is_leaf=lambda leaf: leaf is _missing
        )


class SoftPlusTransform(ParameterTransformation):
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

        wl.pprint(pytree, width=150, short_arrays=False)
        # {
        #   'a': Parameter(value=Array([2.], dtype=float32), transform=SoftPlusTransform()),
        #   'b': Parameter(value=Array([0.1], dtype=float32), transform=SoftPlusTransform())
        # }

        wl.pprint(pytree_t, width=150, short_arrays=False)
        # {
        #   'a': Parameter(value=Array([1.8545866], dtype=float32), transform=SoftPlusTransform()),
        #   'b': Parameter(value=Array([-2.2521687], dtype=float32), transform=SoftPlusTransform())
        # }

        wl.pprint(pytree_tt, width=150, short_arrays=False)
        # {
        #   'a': Parameter(value=Array([2.], dtype=float32), transform=SoftPlusTransform()),
        #   'b': Parameter(value=Array([0.09999998], dtype=float32), transform=SoftPlusTransform())
        # }
    """

    def unwrap(self, parameter: Parameter) -> Parameter:
        # from: https://github.com/danielward27/paramax/blob/main/paramax/utils.py
        """The inverse of the softplus function, checking for positive inputs."""
        parameter = eqx.error_if(
            parameter,
            parameter.value < 0,
            "Expected positive inputs to inv_softplus.",
        )
        value_t = jnp.log(-jnp.expm1(-parameter.value)) + parameter.value
        return eqx.tree_at(lambda p: p.value, parameter, value_t)

    def wrap(self, parameter: Parameter) -> Parameter:
        value_t = jax.nn.softplus(parameter.value)
        return eqx.tree_at(lambda p: p.value, parameter, value_t)
