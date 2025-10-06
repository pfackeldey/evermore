from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import equinox as eqx
import jax
from jaxtyping import PyTree

from evermore.parameters.filter import (
    Filter,
    is_not_frozen,
    is_parameter,
    is_value,
)
from evermore.parameters.mutable import to_arrays, to_refs
from evermore.parameters.parameter import AbstractParameter, V, update_value
from evermore.util import _missing

__all__ = [
    "combine",
    "only",
    "partition",
    "pure",
    "update_value_and_freeze",
    "value_filter_spec",
]


PT = TypeVar("PT", bound=PyTree[AbstractParameter[V]])


def only(tree: PT, *, filter: Filter) -> PT:
    """
    Filters a PyTree to include only leaves that are instances of the specified type.

    Args:
        tree (PT): A PyTree containing various objects, some of which may be instances of the specified type.
        filter (Filter): A callable that checks if an object is of the specified type.

    Returns:
        PT: A new PyTree containing only the instances of the specified type from the original tree.

    Example:

    .. code-block:: python


        import equinox as eqx
        import wadler_lindig as wl
        import evermore as evm

        params = {
            "a": evm.Parameter(1.0),
            "b": 42,
            "c": evm.Parameter(2.0),
        }

        filtered = evm.tree.only(params, filter=evm.filter.is_parameter)
        wl.pprint(filtered, width=150)
        # {
        #     'a': Parameter(raw_value=ValueAttr(value=f32[](jax)), name=None, lower=None, upper=None, prior=None, frozen=False, transform=None, tags=frozenset()),
        #     'b': --,
        #     'c': Parameter(raw_value=ValueAttr(value=f32[](jax)), name=None, lower=None, upper=None, prior=None, frozen=False, transform=None, tags=frozenset())
        # }
    """
    return eqx.filter(tree, filter, replace=_missing, is_leaf=filter.is_leaf)


def pure(tree: PT) -> PT:
    """
    Extracts the raw values from a parameter tree.

    Args:
        params (PT): A tree structure containing parameter objects.

    Returns:
        PT: A tree structure with the same shape as `params`, but with each parameter replaced by its underlying value.

    Example:

    .. code-block:: python

        import equinox as eqx
        import wadler_lindig as wl
        import evermore as evm

        params = {
            "a": evm.Parameter(1.0),
            "b": 42,
            "c": evm.Parameter(2.0),
        }

        pure_values = evm.tree.pure(params)
        wl.pprint(pure_values, short_arrays=False, width=150)
        # {'a': Array(1., dtype=float32), 'b': --, 'c': Array(2., dtype=float32)}
    """
    parameters = only(tree, filter=is_parameter)
    return jax.tree.map(lambda p: p.value, parameters, is_leaf=is_parameter)


def value_filter_spec(tree: PT, *, filter: Filter) -> PT:
    """
    Splits a PyTree of `AbstractParameter` instances into two PyTrees: one containing the values of the parameters
    and the other containing the rest of the PyTree. This is useful for defining which components are to be optimized
    and which to keep fixed during optimization.

    Args:
        tree (PT): A PyTree of `AbstractParameter` instances to be split.
        filter (Filter | None, optional): A filter that defines which parameters are static (frozen).
            If provided, it will be used to determine which parameters are static (frozen) and which are dynamic.

    Returns:
        PT: A PyTree with the same structure as the input, but with boolean values indicating
        which parts of the tree are dynamic (True) and which are static (False).

    Usage:

    .. code-block:: python

        from jaxtyping import Array
        import equinox as eqx
        import evermore as evm

        # define a PyTree of parameters
        params = {
            "a": evm.Parameter(value=1.0),
            "b": evm.Parameter(value=2.0),
        }

        # split the PyTree into dynamic and the static parts
        filter_spec = evm.tree.value_filter_spec(params, filter=evm.filter.is_not_frozen)
        dynamic, static = eqx.partition(params, filter_spec)

        # model's first argument is only the dynamic part of the parameter PyTree!!
        def model(dynamic, static, hists) -> Array:
            # combine the dynamic and static parts of the parameter PyTree
            parameters = evm.tree.combine(dynamic, static)
            assert eqx.tree_equal(params, parameters)
            # use the parameters to calculate the model as usual
            ...
    """
    if not isinstance(filter, Filter):
        msg = f"Expected a Filter, got {filter} ({type(filter)=})"  # type: ignore[unreachable]
        raise ValueError(msg)

    # 1. split by the filter
    left_tree, right_tree = eqx.partition(
        tree,
        filter_spec=filter,
        is_leaf=filter.is_leaf,
    )

    # 2. set the .raw_value attr to True for each parameter from the `left_tree`, rest is False
    value_tree = jax.tree.map(is_value, left_tree, is_leaf=is_value.is_leaf)
    false_tree = jax.tree.map(lambda _: False, right_tree, is_leaf=is_value.is_leaf)

    # 3. combine the two trees to get the final filter spec
    return eqx.combine(value_tree, false_tree, is_leaf=filter.is_leaf)


def partition(tree: PT, *, filter: Filter | None = None) -> tuple[PT, PT]:
    """
    Partitions a PyTree of parameters into two separate PyTrees: one containing the dynamic (optimizable) parts
    and the other containing the static parts.

    This function serves as a shorthand for manually creating a filter specification and then using `eqx.partition`
    to split the parameters.

    Args:
        tree (PT): A PyTree of parameters to be partitioned.
        filter (Filter | None, optional): A filter that defines which parameters are static (frozen).
            If provided, it will be used to determine which parameters are static (frozen) and which are dynamic.

    Returns:
        tuple[PT, PT]: A tuple containing two PyTrees. The first PyTree contains the dynamic parts
        of the parameters, and the second PyTree contains the static parts.

    Example:

    .. code-block:: python

        import equinox as eqx
        import wadler_lindig as wl
        import evermore as evm

        params = {"a": evm.Parameter(1.0), "b": evm.Parameter(2.0, frozen=True)}

        # Verbose:
        filter_spec = evm.tree.value_filter_spec(params, filter=evm.filter.is_not_frozen)
        dynamic, static = eqx.partition(params, filter_spec, replace=evm.util._missing)
        wl.pprint(dynamic, width=150)
        # {
        #     'a': Parameter(raw_value=ValueAttr(value=f32[](jax)), name=None, lower=None, upper=None, prior=None, frozen=--, transform=None, tags=frozenset()),
        #     'b': Parameter(raw_value=ValueAttr(value=--), name=None, lower=None, upper=None, prior=None, frozen=--, transform=None, tags=frozenset())
        # }

        wl.pprint(static, width=150)
        # {
        #     'a': Parameter(raw_value=ValueAttr(value=--), name=None, lower=None, upper=None, prior=None, frozen=False, transform=None, tags=frozenset()),
        #     'b': Parameter(raw_value=ValueAttr(value=f32[](jax)), name=None, lower=None, upper=None, prior=None, frozen=True, transform=None, tags=frozenset())
        # }

        # Short hand:
        dynamic, static = evm.tree.partition(params)
    """
    if filter is None:
        # If no filter is provided, we assume all parameters are dynamic,
        # except those that are marked as frozen.
        filter = is_not_frozen
    return eqx.partition(
        tree,
        filter_spec=value_filter_spec(tree, filter=filter),
        replace=_missing,
    )


def combine(*trees: tuple[PT]) -> PT:
    """
    Combines multiple PyTrees of parameters into a single PyTree.

    For each leaf position, returns the first non-_missing value found among the input trees.
    If all values _missing at a given position, returns _missing for that position.

    Args:
        *trees (PT): One or more PyTrees to be combined.

    Returns:
        PT: A PyTree with the same structure as the inputs, where each leaf is the first non-_missing value found at that position.

    Example:

    .. code-block:: python

        import equinox as eqx
        import wadler_lindig as wl
        import evermore as evm

        params = {"a": evm.Parameter(1.0), "b": evm.Parameter(2.0, frozen=True)}

        dynamic, static = evm.tree.partition(params)
        reconstructed_params = evm.tree.combine(dynamic, static)  # inverse of `partition`
        wl.pprint(reconstructed_params, width=150)
        # {
        #     'a': Parameter(raw_value=ValueAttr(value=f32[](jax)), name=None, lower=None, upper=None, prior=None, frozen=False, transform=None, tags=frozenset()),
        #     'b': Parameter(raw_value=ValueAttr(value=f32[](jax)), name=None, lower=None, upper=None, prior=None, frozen=True, transform=None, tags=frozenset())
        # }

        assert eqx.tree_equal(params, reconstructed_params)
    """

    def _combine(*args):
        for arg in args:
            if arg is not _missing:
                return arg
        return _missing

    return jax.tree.map(_combine, *trees, is_leaf=lambda x: x is _missing)


def update_values(tree: PT, *, values: PT, mask: PyTree[bool] | None = None) -> PT:
    """
    Updates the values of all parameters in the given PyTree.

    This function traverses the input PyTree and, for each parameter, updates its value to the corresponding
    value from the `values` PyTree. Non-parameter leaves are left unchanged.

    Args:
        tree (PT): A PyTree containing parameters to be updated.
        values (PT): A PyTree with the same structure as `tree`, containing the new values for each parameter.
        mask (PyTree[bool] | None, optional): A PyTree with the same structure as `tree`, where `True` indicates that the corresponding
            parameter should be updated, and `False` indicates it should remain unchanged. If `None`, all parameters are updated.

    Returns:
        PT: A new PyTree with the same structure as `tree`, but with each parameter's value updated to the corresponding value from `values`.

    Example:

    .. code-block:: python

        import equinox as eqx
        import wadler_lindig as wl
        import evermore as evm

        params = {
            "a": evm.Parameter(1.0),
            "b": evm.Parameter(42),
            "c": evm.Parameter(2.0),
        }

        new_values = {
            "a": 3.14,
            "b": 123,
            "c": 6.28,
        }

        updated_params = evm.tree.update_values(params, new_values)
        wl.pprint(updated_params, short_arrays=False, width=150)
        # {
        #   'a': Parameter(raw_value=ValueAttr(value=Array(3.14, dtype=float64)), ...),
        #   'b': Parameter(raw_value=ValueAttr(value=Array(123, dtype=int64)), ...),
        #   'c': Parameter(raw_value=ValueAttr(value=Array(6.28, dtype=float64)), ...),
        # }

        # Masked update
        mask = {
            "a": True,
            "b": False,
            "c": True,
        }

        updated_params = evm.tree.update_values(params, new_values, mask)
        wl.pprint(updated_params, short_arrays=False, width=150)
        # {
        #   'a': Parameter(raw_value=ValueAttr(value=Array(3.14, dtype=float64)), ...),
        #   'b': Parameter(raw_value=ValueAttr(value=Array(42, dtype=int64)), ...),
        #   'c': Parameter(raw_value=ValueAttr(value=Array(6.28, dtype=float64)), ...),
        # }
    """
    if mask is None:
        mask = jax.tree.map(lambda _: True, values)

    def _masked_update(
        param: AbstractParameter[V], value: V, mask: bool
    ) -> AbstractParameter[V]:
        if mask:
            return update_value(param, value)
        return param

    return jax.tree.map(
        _masked_update,
        tree,
        values,
        mask,
        is_leaf=is_parameter,
    )


def update_value_and_freeze(tree: PT, *, where: Callable, value: V) -> PT:
    """
    Updates the value of all parameters in the given PyTree and freezes them.

    This function traverses the input PyTree and, for each parameter, updates its value to `value`
    and sets its `frozen` attribute to `True`. Non-parameter leaves are left unchanged.

    Args:
        tree (PT): A PyTree containing parameters to be updated.
        where (Callable): A callable that determines which parameters to update. It should accept a parameter
        value (V): The new value to set for each parameter.

    Returns:
        PT: A new PyTree with the same structure as `tree`, but with each parameter's value updated to `value`
        and frozen.

    Example:

    .. code-block:: python

        import equinox as eqx
        import wadler_lindig as wl
        import evermore as evm

        params = {
            "a": evm.Parameter(1.0),
            "b": 42,
            "c": evm.Parameter(2.0),
        }
        updated_params = evm.tree.update_value_and_freeze(params, lambda t: t["a"], value=3.14)
        wl.pprint(updated_params, short_arrays=False, width=150)
        # {
        #   'a': Parameter(raw_value=ValueAttr(value=Array(3.14, dtype=float64)), ..., frozen=True, ...),
        #   'b': 42,
        #   'c': Parameter(raw_value=ValueAttr(value=Array(2.0, dtype=float64)), ...),
        # }
        assert updated_params["a"].value == 3.14
        assert updated_params["a"].frozen is True

    """
    # First we update the value using mutable arrays
    refs = to_refs(tree)
    param = where(refs)
    if not isinstance(param, AbstractParameter):
        msg = f"Expected a Parameter, got {param} ({type(param)=}) using {where=}"  # type: ignore[unreachable]
        raise ValueError(msg)
    # set the value
    param[...] = value
    tree = to_arrays(refs)
    # Then we freeze the parameter using `eqx.tree_at`
    return eqx.tree_at(lambda t: where(t).frozen, tree, True)
