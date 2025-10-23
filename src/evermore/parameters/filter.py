from __future__ import annotations

import dataclasses
from collections.abc import Hashable

from flax import nnx

from evermore.parameters.parameter import BaseParameter

__all__ = [
    "HasName",
    "HasTags",
    "IsFrozen",
    "is_dynamic_parameter",
    "is_frozen",
    "is_not_frozen",
    "is_parameter",
]


@dataclasses.dataclass(frozen=True)
class IsFrozen:
    """Filter that selects parameters marked as frozen."""

    def __call__(self, path, x):
        del path  # unused
        return hasattr(x, "frozen") and x.frozen


@dataclasses.dataclass(frozen=True)
class HasName:
    """Filter that matches parameters by their ``name`` attribute.

    Attributes:
        name: Required name.
    """

    name: str

    def __call__(self, path, x: BaseParameter):
        del path  # unused
        return hasattr(x, "name") and x.name == self.name


@dataclasses.dataclass(frozen=True)
class HasTags:
    """Filter that checks if a parameter carries a set of tags.

    Attributes:
        tags: Tags that must be a subset of the parameter's tags.
    """

    tags: frozenset[Hashable]

    def __call__(self, path, x: BaseParameter):
        del path  # unused
        return hasattr(x, "tags") and self.tags <= x.tags


is_parameter = nnx.OfType(BaseParameter)
"""Filter that keeps only instances of ``BaseParameter``."""

is_frozen = IsFrozen()
"""Filter that keeps parameters with ``frozen=True``."""

is_not_frozen = nnx.Not(is_frozen)
"""Filter that excludes frozen parameters."""

is_dynamic_parameter = nnx.All(is_parameter, is_not_frozen)
