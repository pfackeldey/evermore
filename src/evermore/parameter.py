from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree

from evermore.custom_types import PDFLike
from evermore.pdf import Normal, Poisson
from evermore.util import filter_tree_map

if TYPE_CHECKING:
    from evermore.modifier import Modifier


__all__ = [
    "Parameter",
    "NormalParameter",
    "is_parameter",
    "params_map",
    "value_filter_spec",
    "partition",
    "sample",
    "correlate",
]


def __dir__():
    return __all__


class Parameter(eqx.Module):
    """
    Implementation of a general Parameter class. The class is used to define the parameters of a statistical model.
    Key is the value attribute, which holds the actual value of the parameter. In additon,
    the lower and upper attributes define the boundaries of the parameter. The prior attribute
    is used to define the prior distribution of the parameter. The frozen attribute is used to
    freeze the parameter during optimization.

    Usage:

    .. code-block:: python

        import evermore as evm

        simple_param = evm.Parameter(value=1.0)
        bounded_param = evm.Parameter(value=1.0, lower=0.0, upper=2.0)
        constrained_parameter = evm.Parameter(value=1.0, prior=evm.pdf.Normal(mean=1.0, width=0.1))
        frozen_parameter = evm.Parameter(value=1.0, frozen=True)
    """

    value: Array = eqx.field(converter=jnp.atleast_1d, default=0.0)
    name: str | None = eqx.field(static=True, default=None)
    lower: Array = eqx.field(converter=jnp.atleast_1d, default=-jnp.inf)
    upper: Array = eqx.field(converter=jnp.atleast_1d, default=jnp.inf)
    prior: PDFLike | None = eqx.field(default=None)
    frozen: bool = eqx.field(static=True, default=False)

    @property
    def boundary_constraint(self) -> Array:
        return jnp.where(
            (self.value < self.lower) | (self.value > self.upper),
            jnp.inf,
            0,
        )

    def scale(self, slope: ArrayLike = 1.0, offset: ArrayLike = 0.0) -> Modifier:
        from evermore.effect import Linear
        from evermore.modifier import Modifier

        return Modifier(
            parameter=self,
            effect=Linear(slope=slope, offset=offset),
        )


class NormalParameter(Parameter):
    prior: PDFLike | None = Normal(mean=0.0, width=1.0)

    def scale_log(self, up: ArrayLike, down: ArrayLike) -> Modifier:
        from evermore.effect import AsymmetricExponential
        from evermore.modifier import Modifier

        return Modifier(parameter=self, effect=AsymmetricExponential(up=up, down=down))

    def morphing(self, up_template: Array, down_template: Array) -> Modifier:
        from evermore.effect import VerticalTemplateMorphing
        from evermore.modifier import Modifier

        return Modifier(
            parameter=self,
            effect=VerticalTemplateMorphing(
                up_template=up_template, down_template=down_template
            ),
        )


def is_parameter(leaf: Any) -> bool:
    return isinstance(leaf, Parameter)


params_map = partial(filter_tree_map, filter=is_parameter)


def value_filter_spec(tree: PyTree) -> PyTree:
    """
    Used to split a PyTree of evm.Parameters into two PyTrees: one containing the values of the parameters
    and the other containing the rest of the PyTree. This is useful for defining which components are to be optimized
    and which to keep fixed during optimization.

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

        # model's first argument is only the diffable part of the parameter Pytree!!
        def model(diffable, static, hists) -> Array:
            # combine the diffable and static parts of the parameter PyTree
            parameters = eqx.combine(diffable, static)
            assert parameters == params
            # use the parameters to calculate the model as usual
            ...
    """
    # 1. set the filter_spec to False for all non-static leaves
    filter_spec = jtu.tree_map(lambda _: False, tree)

    # 2. set the filter_spec to True for each parameter value
    def _replace_value(leaf: Any) -> Any:
        if isinstance(leaf, Parameter):
            leaf = eqx.tree_at(lambda p: p.value, leaf, not leaf.frozen)
        return leaf

    return jtu.tree_map(_replace_value, filter_spec, is_leaf=is_parameter)


def partition(tree: PyTree) -> tuple[PyTree, PyTree]:
    """
    Short hand for:

    .. code-block:: python

        import evermore as evm

        # Verbose:
        filter_spec = evm.parameter.value_filter_spec(params)
        diffable, static = eqx.partition(params, filter_spec)

        # Short hand:
        diffable, static = evm.parameter.partition(params)
    """
    return eqx.partition(tree, filter_spec=value_filter_spec(tree))


def sample(tree: PyTree, key: PRNGKeyArray) -> PyTree:
    """
    Sample from the individual prior (no correlations taken into account!) of the parameters in the PyTree.
    See examples/toy_generation.py for an example.
    """
    # partition the tree into parameters and the rest
    params_tree, rest_tree = eqx.partition(tree, is_parameter, is_leaf=is_parameter)
    params_structure = jax.tree_util.tree_structure(params_tree)
    n_params = params_structure.num_leaves  # type: ignore[attr-defined]

    keys = jax.random.split(key, n_params)
    keys_tree = jax.tree_util.tree_unflatten(params_structure, keys)

    def _sample(param: Parameter, key: Parameter) -> Array:
        if isinstance(param.prior, PDFLike):
            pdf = param.prior
            pdf = cast(PDFLike, pdf)

            # sample new value from the prior pdf
            sampled_value = pdf.sample(key.value)

            # TODO: make this compatible with externally provided Poisson PDFs
            if isinstance(pdf, Poisson):
                sampled_value = (sampled_value / pdf.lamb) - 1
        else:
            assert param.prior is None, f"Unknown prior type: {param.prior}."
            if not jnp.isfinite(param.lower) and not jnp.isfinite(param.upper):
                msg = f"Can't sample uniform from {param} (no given prior), because of non-finite bounds. "
                msg += "Please provide finite bounds."
                raise ValueError(msg)
            sampled_value = jax.random.uniform(
                key.value,
                shape=param.value.shape,
                minval=param.lower,
                maxval=param.upper,
            )

        # replace the sampled parameter value and return new parameter
        return eqx.tree_at(lambda p: p.value, param, sampled_value)

    # sample for each parameter
    sampled_params_tree = jtu.tree_map(
        _sample, params_tree, keys_tree, is_leaf=is_parameter
    )

    # combine the sampled parameters with the rest of the model and return it
    return eqx.combine(sampled_params_tree, rest_tree, is_leaf=is_parameter)


def correlate(*parameters: Parameter) -> tuple[Parameter, ...]:
    """
    Correlate parameters by sharing the value of the *first* given parameter.

    It is preferred to just use the same parameter if possible, this function should be used if that is not doable.

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
        import jax.tree_util as jtu


        class Params(NamedTuple):
            mu: evm.Parameter
            syst: evm.NormalParameter

        params = Params(mu=evm.Parameter(1.0), syst=evm.NormalParameter(0.0))

        def model(params: Params):
            flat_params, tree_def = jtu.tree_flatten(params, evm.parameter.is_parameter)

            # correlate the parameters
            correlated_flat_params = evm.parameter.correlate(*flat_params)
            correlated_params = jtu.tree_unflatten(tree_def, correlated_flat_params)

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
