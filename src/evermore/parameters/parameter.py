from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import jax
from jaxtyping import Array, ArrayLike, Float, PyTree

from evermore.pdf import PDF, Normal
from evermore.util import _missing, filter_tree_map, float_array
from evermore.visualization import SupportsTreescope

if TYPE_CHECKING:
    from evermore.binned.modifier import Modifier
    from evermore.parameters.transform import ParameterTransformation

__all__ = [
    "NormalParameter",
    "Parameter",
    "correlate",
    "is_parameter",
    "partition",
    "replace_value",
    "value_filter_spec",
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
        prior (PDF | None): The prior distribution of the parameter.
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

    value: Float[Array, "..."] = eqx.field(converter=float_array, default=0.0)  # noqa: UP037
    name: str | None = eqx.field(default=None)
    lower: Float[Array, "..."] | None = eqx.field(default=None)  # noqa: UP037
    upper: Float[Array, "..."] | None = eqx.field(default=None)  # noqa: UP037
    prior: PDF | None = eqx.field(default=None)
    frozen: bool = eqx.field(default=False)
    transform: ParameterTransformation | None = eqx.field(default=None)

    def __check_init__(self):
        # runtime check to be sure
        if self.prior is not None and not isinstance(self.prior, PDF):
            msg = f"Prior must be a PDF object for a constrained Parameter (or 'None' for an unconstrained one), got {self.prior=} ({type(self.prior)=})"  # type: ignore[unreachable]
            raise ValueError(msg)

    def scale(self, slope: ArrayLike = 1.0, offset: ArrayLike = 0.0) -> Modifier:
        """
        Applies a linear scaling effect to the parameter.

        Args:
            slope (ArrayLike, optional): The slope of the linear scaling. Defaults to 1.0.
            offset (ArrayLike, optional): The offset of the linear scaling. Defaults to 0.0.

        Returns:
            Modifier: A Modifier instance with the linear scaling effect applied.
        """
        from evermore.binned.effect import Linear
        from evermore.binned.modifier import Modifier

        return Modifier(
            parameter=self,
            effect=Linear(slope=slope, offset=offset),  # type: ignore[arg-type]
        )


class NormalParameter(Parameter):
    """
    A specialized Parameter class with a Normal prior distribution.

    This class extends the general Parameter class by setting a default Normal prior distribution.
    It also provides additional methods for scaling and morphing the parameter.

    Attributes:
        prior (PDF | None): The prior distribution of the parameter, defaulting to a Normal distribution with mean 0.0 and width 1.0.
    """

    prior: PDF | None = eqx.field(default_factory=lambda: Normal(mean=0.0, width=1.0))  # type: ignore[arg-type]

    def scale_log(self, up: Float[Array, "..."], down: Float[Array, "..."]) -> Modifier:  # noqa: UP037
        """
        Applies an asymmetric exponential scaling to the parameter.

        Args:
            up (Float[Array, "..."]): The scaling factor for the upward direction.
            down (Float[Array, "..."]): The scaling factor for the downward direction.

        Returns:
            Modifier: A Modifier instance with the asymmetric exponential effect applied.
        """
        from evermore.binned.effect import AsymmetricExponential
        from evermore.binned.modifier import Modifier

        return Modifier(parameter=self, effect=AsymmetricExponential(up=up, down=down))  # type: ignore[arg-type]

    def morphing(
        self,
        up_template: Float[Array, "..."],  # noqa: UP037
        down_template: Float[Array, "..."],  # noqa: UP037
    ) -> Modifier:
        """
        Applies vertical template morphing to the parameter.

        Args:
            up_template (Float[Array, "..."]): The template for the upward shift.
            down_template (Float[Array, "..."]): The template for the downward shift.

        Returns:
            Modifier: A Modifier instance with the vertical template morphing effect applied.
        """
        from evermore.binned.effect import VerticalTemplateMorphing
        from evermore.binned.modifier import Modifier

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


def replace_value(
    param: Parameter,
    value: Float[Array, "..."],  # noqa: UP037
) -> Parameter:
    return eqx.tree_at(
        lambda p: p.value,
        param,
        value,
        is_leaf=lambda leaf: leaf is _missing,
    )


def value_filter_spec(tree: _ParamsTree) -> _ParamsTree:
    """
    Splits a PyTree of `evm.Parameter` instances into two PyTrees: one containing the values of the parameters
    and the other containing the rest of the PyTree. This is useful for defining which components are to be optimized
    and which to keep fixed during optimization.

    Args:
        tree (_ParamsTree): A PyTree of `evm.Parameter` instances to be split.

    Returns:
        _ParamsTree: A PyTree with the same structure as the input, but with boolean values indicating
        which parts of the tree are dynamic (True) and which are static (False).

    Usage:

    .. code-block:: python

        from jaxtyping import Array
        import evermore as evm

        # define a PyTree of parameters
        params = {
            "a": evm.Parameter(value=1.0),
            "b": evm.Parameter(value=2.0),
        }

        # split the PyTree into dynamic and the static parts
        filter_spec = evm.parameter.value_filter_spec(params)
        dynamic, static = eqx.partition(params, filter_spec)

        # model's first argument is only the dynamic part of the parameter PyTree!!
        def model(dynamic, static, hists) -> Array:
            # combine the dynamic and static parts of the parameter PyTree
            parameters = eqx.combine(dynamic, static)
            assert parameters == params
            # use the parameters to calculate the model as usual
            ...
    """
    # 1. set the filter_spec to False for all (non-static) leaves
    filter_spec = jax.tree.map(lambda _: False, tree)

    # 2. set the filter_spec to True for each parameter value,
    # and _only_ the .value, because we don't want do optimize against anything else!
    def _replace_value(filter_leaf: Any, tree_leaf: Any) -> Any:
        if isinstance(filter_leaf, Parameter):
            filter_leaf = eqx.tree_at(
                lambda fl: fl.value,
                filter_leaf,
                not tree_leaf.frozen,
                is_leaf=lambda leaf: leaf is _missing,
            )
        return filter_leaf

    return jax.tree.map(_replace_value, filter_spec, tree, is_leaf=is_parameter)


def partition(tree: _ParamsTree) -> tuple[_ParamsTree, _ParamsTree]:
    """
    Partitions a PyTree of parameters into two separate PyTrees: one containing the dynamic (optimizable) parts
    and the other containing the static parts.

    This function serves as a shorthand for manually creating a filter specification and then using `eqx.partition`
    to split the parameters.

    Args:
        tree (_ParamsTree): A PyTree of parameters to be partitioned.

    Returns:
        tuple[_ParamsTree, _ParamsTree]: A tuple containing two PyTrees. The first PyTree contains the dynamic parts
        of the parameters, and the second PyTree contains the static parts.

    Example:

    .. code-block:: python

        import evermore as evm

        params = {"a": evm.Parameter(1.0), "b": evm.Parameter(2.0, frozen=True)}

        # Verbose:
        filter_spec = evm.parameter.value_filter_spec(params)
        dynamic, static = eqx.partition(params, filter_spec, replace=evm.util._missing)
        print(dynamic)
        # >> {'a': Parameter(value=f32[1]), 'b': Parameter(value=--, frozen=True)}

        print(static)
        # >> {'a': Parameter(value=--), 'b': Parameter(value=f32[1], frozen=True)}

        # Short hand:
        dynamic, static = evm.parameter.partition(params)
    """
    return eqx.partition(tree, filter_spec=value_filter_spec(tree), replace=_missing)


def combine(*trees: tuple[_ParamsTree]) -> _ParamsTree:
    """
    Combines multiple PyTrees of parameters into a single PyTree.

    For each leaf position, returns the first non-_missing value found among the input trees.
    If all values _missing at a given position, returns _missing for that position.

    Args:
        *trees (_ParamsTree): One or more PyTrees to be combined.

    Returns:
        _ParamsTree: A PyTree with the same structure as the inputs, where each leaf is the first non-_missing value found at that position.

    Example:

    .. code-block:: python

        import evermore as evm

        params = {"a": evm.Parameter(1.0), "b": evm.Parameter(2.0, frozen=True)}

        dynamic, static = evm.parameter.partition(params)
        reconstructed_params = evm.parameter.combine(dynamic, static)  # inverse of `partition`
        print(reconstructed_params)
        # >> {"a": evm.Parameter(1.0), "b": evm.Parameter(2.0)}
    """

    def _combine(*args):
        for arg in args:
            if arg is not _missing:
                return arg
        return _missing

    return jax.tree.map(_combine, *trees, is_leaf=lambda x: x is _missing)


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

        def where(ps: tuple[Parameter, Parameter]) -> Float[Array, "..."]:  # noqa: UP037
            return ps[1].value

        def get(ps: tuple[Parameter, Parameter]) -> Float[Array, "..."]:  # noqa: UP037
            return ps[0].value

        shared = eqx.nn.Shared(ps, where, get)
        return shared()

    correlated = [first]
    for p in rest:
        # is this error really needed? shouldn't it be safe to broadcast here?
        if p.value.shape != first.value.shape:
            msg = f"Can't correlate parameters {first} and {p}! Must have the same shape, got {first.value.shape} and {p.value.shape}."
            raise ValueError(msg)
        _, p_corr = _correlate(p)
        correlated.append(p_corr)
    return tuple(correlated)
