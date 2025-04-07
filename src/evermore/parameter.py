from __future__ import annotations

import abc
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree

from evermore.custom_types import PDFLike
from evermore.pdf import Normal, Poisson
from evermore.util import atleast_1d_float_array, filter_tree_map
from evermore.visualization import SupportsTreescope

if TYPE_CHECKING:
    from evermore.modifier import Modifier


__all__ = [
    "MinuitTransform",
    "NormalParameter",
    "Parameter",
    "ParameterTransformation",
    "SoftPlusTransform",
    "correlate",
    "is_parameter",
    "partition",
    "sample",
    "unwrap",
    "value_filter_spec",
    "wrap",
]


def __dir__():
    return __all__


class Parameter(eqx.Module, SupportsTreescope):
    """
    A general Parameter class for defining the parameters of a statistical model.

    Attributes:
        value (Array): The actual value of the parameter.
        name (str | None): An optional name for the parameter.
        lower (Array | None): The lower boundary of the parameter.
        upper (Array | None): The upper boundary of the parameter.
        prior (PDFLike | None): The prior distribution of the parameter.
        frozen (bool): Indicates if the parameter is frozen during optimization.
        transform (ParameterTransformation | None): An optional transformation applied to the parameter.

    Usage:

    .. code-block:: python

        import evermore as evm

        simple_param = evm.Parameter(value=1.0)
        bounded_param = evm.Parameter(value=1.0, lower=0.0, upper=2.0)
        constrained_parameter = evm.Parameter(
            value=1.0, prior=evm.pdf.Normal(mean=1.0, width=0.1)
        )
        frozen_parameter = evm.Parameter(value=1.0, frozen=True)
    """

    value: Array = eqx.field(converter=atleast_1d_float_array, default=0.0)
    name: str | None = eqx.field(static=True, default=None)
    lower: Array | None = eqx.field(default=None)
    upper: Array | None = eqx.field(default=None)
    prior: PDFLike | None = eqx.field(default=None)
    frozen: bool = eqx.field(static=True, default=False)
    transform: ParameterTransformation | None = eqx.field(default=None)

    def scale(self, slope: ArrayLike = 1.0, offset: ArrayLike = 0.0) -> Modifier:
        """
        Applies a linear scaling effect to the parameter.

        Args:
            slope (ArrayLike, optional): The slope of the linear scaling. Defaults to 1.0.
            offset (ArrayLike, optional): The offset of the linear scaling. Defaults to 0.0.

        Returns:
            Modifier: A Modifier instance with the linear scaling effect applied.
        """
        from evermore.effect import Linear
        from evermore.modifier import Modifier

        return Modifier(
            parameter=self,
            effect=Linear(slope=slope, offset=offset),
        )


class NormalParameter(Parameter):
    """
    A specialized Parameter class with a Normal prior distribution.

    This class extends the general Parameter class by setting a default Normal prior distribution.
    It also provides additional methods for scaling and morphing the parameter.

    Attributes:
        prior (PDFLike | None): The prior distribution of the parameter, defaulting to a Normal distribution with mean 0.0 and width 1.0.
    """

    prior: PDFLike | None = Normal(mean=0.0, width=1.0)

    def scale_log(self, up: ArrayLike, down: ArrayLike) -> Modifier:
        """
        Applies an asymmetric exponential scaling to the parameter.

        Args:
            up (ArrayLike): The scaling factor for the upward direction.
            down (ArrayLike): The scaling factor for the downward direction.

        Returns:
            Modifier: A Modifier instance with the asymmetric exponential effect applied.
        """
        from evermore.effect import AsymmetricExponential
        from evermore.modifier import Modifier

        return Modifier(parameter=self, effect=AsymmetricExponential(up=up, down=down))

    def morphing(self, up_template: Array, down_template: Array) -> Modifier:
        """
        Applies vertical template morphing to the parameter.

        Args:
            up_template (Array): The template for the upward shift.
            down_template (Array): The template for the downward shift.

        Returns:
            Modifier: A Modifier instance with the vertical template morphing effect applied.
        """
        from evermore.effect import VerticalTemplateMorphing
        from evermore.modifier import Modifier

        return Modifier(
            parameter=self,
            effect=VerticalTemplateMorphing(
                up_template=up_template, down_template=down_template
            ),
        )


def is_parameter(leaf: Any) -> bool:
    """
    Checks if the given leaf is an instance of the Parameter class.

    Args:
        leaf (Any): The object to check.

    Returns:
        bool: True if the leaf is an instance of Parameter, False otherwise.
    """
    return isinstance(leaf, Parameter)


_params_map = partial(filter_tree_map, filter=is_parameter)


_ParamsTree = TypeVar("_ParamsTree", bound=PyTree[Parameter])


def value_filter_spec(tree: _ParamsTree) -> _ParamsTree:
    """
    Splits a PyTree of `evm.Parameter` instances into two PyTrees: one containing the values of the parameters
    and the other containing the rest of the PyTree. This is useful for defining which components are to be optimized
    and which to keep fixed during optimization.

    Args:
        tree (_ParamsTree): A PyTree of `evm.Parameter` instances to be split.

    Returns:
        _ParamsTree: A PyTree with the same structure as the input, but with boolean values indicating
        which parts of the tree are diffable (True) and which are static (False).

    Usage:

    .. code-block:: python

        from jaxtyping import Array
        import evermore as evm

        # define a PyTree of parameters
        params = {
            "a": evm.Parameter(value=1.0),
            "b": evm.Parameter(value=2.0),
        }

        # split the PyTree into diffable and the static parts
        filter_spec = evm.parameter.value_filter_spec(params)
        diffable, static = eqx.partition(params, filter_spec)

        # model's first argument is only the diffable part of the parameter PyTree!!
        def model(diffable, static, hists) -> Array:
            # combine the diffable and static parts of the parameter PyTree
            parameters = eqx.combine(diffable, static)
            assert parameters == params
            # use the parameters to calculate the model as usual
            ...
    """
    # 1. set the filter_spec to False for all non-static leaves
    filter_spec = jax.tree.map(lambda _: False, tree)

    # 2. set the filter_spec to True for each parameter value
    def _replace_value(leaf: Any) -> Any:
        if isinstance(leaf, Parameter):
            leaf = eqx.tree_at(lambda p: p.value, leaf, not leaf.frozen)
        return leaf

    return jax.tree.map(_replace_value, filter_spec, is_leaf=is_parameter)


def partition(tree: _ParamsTree) -> tuple[_ParamsTree, _ParamsTree]:
    """
    Partitions a PyTree of parameters into two separate PyTrees: one containing the diffable (optimizable) parts
    and the other containing the static parts.

    This function serves as a shorthand for manually creating a filter specification and then using `eqx.partition`
    to split the parameters.

    Args:
        tree (_ParamsTree): A PyTree of parameters to be partitioned.

    Returns:
        tuple[_ParamsTree, _ParamsTree]: A tuple containing two PyTrees. The first PyTree contains the diffable parts
        of the parameters, and the second PyTree contains the static parts.

    Example:

    .. code-block:: python

        import evermore as evm

        # Verbose:
        filter_spec = evm.parameter.value_filter_spec(params)
        diffable, static = eqx.partition(params, filter_spec)

        # Short hand:
        diffable, static = evm.parameter.partition(params)
    """
    return eqx.partition(tree, filter_spec=value_filter_spec(tree))


def sample(tree: _ParamsTree, key: PRNGKeyArray) -> _ParamsTree:
    """
    Samples from the individual prior distributions of the parameters in the given PyTree.
    Note that no correlations between parameters are taken into account during sampling.

    Args:
        tree (_ParamsTree): A PyTree of parameters from which to sample.
        key (PRNGKeyArray): A JAX random key used for generating random samples.

    Returns:
        _ParamsTree: A new PyTree with the parameters sampled from their respective prior distributions.

    Example:
        See examples/toy_generation.py for an example usage.
    """
    # Partition the tree into parameters and the rest
    params_tree, rest_tree = eqx.partition(tree, is_parameter, is_leaf=is_parameter)
    params_structure = jax.tree.structure(params_tree)
    n_params = params_structure.num_leaves  # type: ignore[attr-defined]

    keys = jax.random.split(key, n_params)
    keys_tree = jax.tree.unflatten(params_structure, keys)

    def _sample(param: Parameter, key: Parameter) -> Array:
        if isinstance(param.prior, PDFLike):
            pdf = param.prior
            pdf = cast(PDFLike, pdf)

            # Sample new value from the prior pdf
            sampled_value = pdf.sample(key.value)

            # TODO: Make this compatible with externally provided Poisson PDFs
            if isinstance(pdf, Poisson):
                sampled_value = (sampled_value / pdf.lamb) - 1
        else:
            assert param.prior is None, f"Unknown prior type: {param.prior}."
            msg = f"Can't sample uniform from {param} (no given prior). "
            param = eqx.error_if(
                param, ~jnp.isfinite(param.lower), msg + "No lower bound given."
            )
            param = eqx.error_if(
                param, ~jnp.isfinite(param.upper), msg + "No upper bound given."
            )
            sampled_value = jax.random.uniform(
                key.value,
                shape=param.value.shape,
                minval=param.lower,
                maxval=param.upper,
            )

        # Replace the sampled parameter value and return new parameter
        return eqx.tree_at(lambda p: p.value, param, sampled_value)

    # Sample for each parameter
    sampled_params_tree = jax.tree.map(
        _sample, params_tree, keys_tree, is_leaf=is_parameter
    )

    # Combine the sampled parameters with the rest of the model and return it
    return eqx.combine(sampled_params_tree, rest_tree, is_leaf=is_parameter)


def correlate(*parameters: Parameter) -> tuple[Parameter, ...]:
    """
    Correlate parameters by sharing the value of the *first* given parameter.

    It is preferred to just use the same parameter if possible, this function should be used if that is not doable.

    Args:
        *parameters (Parameter): A variable number of Parameter instances to be correlated.

    Returns:
        tuple[Parameter, ...]: A tuple of correlated Parameter instances.

    Example:

    .. code-block:: python

        from jaxtyping import PyTree
        import evermore as evm

        p1 = evm.Parameter(value=1.0)
        p2 = evm.Parameter(value=0.0)
        p3 = evm.Parameter(value=0.5)


        def model(*parameters: PyTree[evm.Parameter]):
            # correlate them inside the model
            p1, p2, p3 = evm.parameter.correlate(*parameters)

            # now p1, p2, p3 are correlated, i.e., they share the same value
            assert p1.value == p2.value == p3.value


        # use the model
        model(p1, p2, p3)

        # More general case of correlating any PyTree of parameters
        from typing import NamedTuple


        class Params(NamedTuple):
            mu: evm.Parameter
            syst: evm.NormalParameter


        params = Params(mu=evm.Parameter(1.0), syst=evm.NormalParameter(0.0))


        def model(params: Params):
            flat_params, tree_def = jax.tree.flatten(params, evm.parameter.is_parameter)

            # correlate the parameters
            correlated_flat_params = evm.parameter.correlate(*flat_params)
            correlated_params = jax.tree.unflatten(tree_def, correlated_flat_params)

            # now correlated_params.mu and correlated_params.syst are correlated, i.e., they share the same value
            assert correlated_params.mu.value == correlated_params.syst.value


        # use the model
        model(params)
    """

    first, *rest = parameters

    def _correlate(parameter: Parameter) -> tuple[Parameter, Parameter]:
        ps = (first, parameter)

        def where(ps: tuple[Parameter, Parameter]) -> Array:
            return ps[1].value

        def get(ps: tuple[Parameter, Parameter]) -> Array:
            return ps[0].value

        shared = eqx.nn.Shared(ps, where, get)
        return shared()

    correlated = [first]
    for p in rest:
        if p.value.shape != first.value.shape:
            msg = f"Can't correlate parameters {first} and {p}! Must have the same shape, got {first.value.shape} and {p.value.shape}."
            raise ValueError(msg)
        _, p_corr = _correlate(p)
        correlated.append(p_corr)
    return tuple(correlated)


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


            minuit_transform = evm.parameter.MinuitTransform()
            pytree = {
                "a": evm.Parameter(2.0, lower=-0.1, upper=2.2, transform=minuit_transform),
                "b": evm.Parameter(0.1, lower=0.0, upper=1.1, transform=minuit_transform),
            }

            # unwrap (or "transform")
            pytree_t = evm.parameter.unwrap(pytree)
            # wrap back (or "inverse transform")
            pytree_tt = evm.parameter.wrap(pytree_t)

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
        # check for finite boundaries
        error_msg = f"Bounds of {parameter} must be finite, got {parameter.lower=}, {parameter.upper=}."
        parameter = eqx.error_if(
            parameter,
            ~jnp.isfinite(parameter.lower),
            error_msg,
        )
        return eqx.error_if(
            parameter,
            ~jnp.isfinite(parameter.upper),
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
        return eqx.tree_at(lambda p: p.value, parameter, value_t)

    def wrap(self, parameter: Parameter) -> Parameter:
        # short-cut
        if parameter.lower is None and parameter.upper is None:
            return parameter

        parameter = self._check(parameter)
        # this formula turns "internal" parameter values into "external" values
        value_t = parameter.lower + (parameter.upper - parameter.lower) / 2 * (  # type: ignore[operator]
            jnp.sin(parameter.value) + 1
        )
        return eqx.tree_at(lambda p: p.value, parameter, value_t)


class SoftPlusTransform(ParameterTransformation):
    """
    Applies the softplus transformation to parameters, projecting them from real space (R) to positive space (R+).
    This transformation is useful for enforcing the positivity of parameters and does not require lower or upper boundaries.

    Use `unwrap` to transform parameters into the unconstrained real space and `wrap` to transform them back into the positive real space.

    Example:

    .. code-block:: python

        import evermore as evm
        import wadler_lindig as wl


        enforce_positivity = evm.parameter.SoftPlusTransform()
        pytree = {
            "a": evm.Parameter(2.0, transform=enforce_positivity),
            "b": evm.Parameter(0.1, transform=enforce_positivity),
        }

        # unwrap (or "transform")
        pytree_t = evm.parameter.unwrap(pytree)
        # wrap back (or "inverse transform")
        pytree_tt = evm.parameter.wrap(pytree_t)

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
