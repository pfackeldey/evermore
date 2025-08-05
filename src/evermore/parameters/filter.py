from __future__ import annotations

import abc
import dataclasses
from collections.abc import Hashable
from typing import Any

from evermore.parameters.parameter import AbstractParameter, ValueAttr

__all__ = [
    "Filter",
    "HasName",
    "HasTags",
    "IsFrozen",
    "IsParam",
    "OfType",
    "ParameterFilter",
    "is_frozen",
    "is_not_frozen",
    "is_parameter",
    "is_value",
]


@dataclasses.dataclass(frozen=True)
class Filter(abc.ABC):
    """
    Base class for filters that can be used to filter PyTrees with the evm.tree.* module.
    """

    @abc.abstractmethod
    def __call__(self, x: Any) -> bool: ...

    @abc.abstractmethod
    def is_leaf(self, x: Any) -> bool: ...


@dataclasses.dataclass(frozen=True)
class Not(Filter):
    """
    A filter class that inverts the result of another filter.

    Attributes:
        filter (Filter): The filter whose result will be negated.
    """

    filter: Filter

    def __call__(self, x: Any) -> bool:
        """
        Makes the instance callable and returns the negation of the filter result for the given input.

        Args:
            x (Any): The input to be evaluated by the filter.

        Returns:
            bool: True if the filter returns False for the input, otherwise False.
        """
        return not self.filter(x)

    def is_leaf(self, x: Any) -> bool:
        return self.filter.is_leaf(x)


@dataclasses.dataclass(frozen=True)
class OfType(Filter):
    """
    A filter that checks if a value is of a specified type.

    Attributes:
        type (type): The type to check against.
    """

    type: type

    def __call__(self, x: Any):
        """
        Check if the input object is an instance of the specified type.

        Args:
            x (Any): The object to check.

        Returns:
            bool: True if x is an instance of self.type, False otherwise.
        """
        return isinstance(x, self.type)

    def is_leaf(self, x: Any) -> bool:
        return self(x)


@dataclasses.dataclass(frozen=True)
class ParameterFilter(Filter):
    def is_leaf(self, x: Any) -> bool:
        return is_parameter(x)


@dataclasses.dataclass(frozen=True)
class IsParam(ParameterFilter):
    """
    A parameter filter that matches a specific parameter instance.

    Attributes:
        param (AbstractParameter): The parameter instance to match.
    """

    param: AbstractParameter

    def __post_init__(self):
        if not isinstance(self.param, AbstractParameter):
            msg = f"Expected an AbstractParameter, got {type(self.param).__name__}"  # type: ignore[unreachable]
            raise TypeError(msg)

    def __call__(self, x: AbstractParameter) -> bool:
        """
        Checks if the given parameter is the same instance as the stored parameter.

        Args:
            x (AbstractParameter): The parameter to compare.

        Returns:
            bool: True if `x` is the same instance as `self.param`, False otherwise.
        """
        return x is self.param


@dataclasses.dataclass(frozen=True)
class HasName(ParameterFilter):
    """
    A filter that matches parameters by their name.

    Attributes:
        name (str): The name to match against the parameter's name.
    """

    name: str

    def __call__(self, x: AbstractParameter):
        """
        Compares the name attribute of this object with that of the given AbstractParameter.

        Args:
            x (AbstractParameter): The parameter to compare against.

        Returns:
            bool: True if the names are equal, False otherwise.
        """
        return self.name == x.name


@dataclasses.dataclass(frozen=True)
class HasTags(ParameterFilter):
    """
    A filter that checks if a parameter has all specified tags.

    Attributes:
        tags (frozenset[Hashable]): The set of tags to check for.
    """

    tags: frozenset[Hashable]

    def __call__(self, x: AbstractParameter) -> bool:
        """
        Determines if the tags of this filter are a subset of the tags of the given AbstractParameter.

        Args:
            x (AbstractParameter): The parameter to check against.

        Returns:
            bool: True if all tags in this filter are present in x.tags, False otherwise.
        """
        return self.tags <= x.tags


@dataclasses.dataclass(frozen=True)
class IsFrozen(ParameterFilter):
    """
    A filter that checks if a parameter is frozen.
    """

    def __call__(self, x: AbstractParameter) -> bool:
        """
        Checks if the given AbstractParameter instance is frozen.

        Args:
            x (AbstractParameter): The parameter to check.

        Returns:
            bool: True if the parameter is frozen, False otherwise.
        """
        return x.frozen


is_parameter = OfType(type=AbstractParameter)
"""
A filter that checks if a value is an instance of AbstractParameter.

Example:

    .. code-block:: python

        import evermore as evm

        params = {
            "a": evm.Parameter(value=1.0),
            "b": 42,
            "c": evm.Parameter(value=2.0),
        }

        filtered_params = evm.tree.only(params, filter=evm.filter.is_parameter)
"""

is_value = OfType(type=ValueAttr)
"""
A filter that checks if a value is an instance of ValueAttr.

Example:

    .. code-block:: python

        import evermore as evm

        params = {
            "a": evm.Parameter(value=1.0),
            "b": 42,
            "c": evm.Parameter(value=2.0),
        }

        filtered_params = evm.tree.only(params, filter=evm.filter.is_value)
"""

is_frozen = IsFrozen()
"""
A filter that checks if a parameter is frozen.

Example:

    .. code-block:: python

        import evermore as evm

        params = {
            "a": evm.Parameter(value=1.0, frozen=True),
            "b": 42,
            "c": evm.Parameter(value=2.0),
        }

        filtered_params = evm.tree.only(params, filter=evm.filter.is_frozen)
"""

is_not_frozen = Not(is_frozen)
"""
A filter that checks if a parameter is not frozen.

Example:

    .. code-block:: python

        import evermore as evm

        params = {
            "a": evm.Parameter(value=1.0, frozen=True),
            "b": 42,
            "c": evm.Parameter(value=2.0),
        }

        filtered_params = evm.tree.only(params, filter=evm.filter.is_not_frozen)
"""
