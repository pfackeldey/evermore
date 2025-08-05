from __future__ import annotations

import typing as tp

import equinox as eqx
import jax
from jax._src.state.types import AbstractRef  # noqa: PLC2701
from jax.experimental import MutableArray, mutable_array
from jaxtyping import PyTree


def _is_mutable_array(x) -> tp.TypeGuard[MutableArray]:
    return isinstance(x, jax.Array | AbstractRef | MutableArray) and isinstance(
        jax.typeof(x), AbstractRef | MutableArray
    )


def to_refs(vals: PyTree) -> PyTree:
    """
    Recursively converts all mutable array-like elements within a parameter tree to their mutable counterparts.

    Args:
        vals (PyTree): The parameter tree to process.

    Returns:
        PyTree: A new parameter tree where all mutable array-like elements have been converted using `mutable_array`.
    """
    return jax.tree.map(lambda x: mutable_array(x) if eqx.is_array(x) else x, vals)


def to_arrays(vals: PyTree) -> PyTree:
    """
    Converts all mutable arrays within the given parameter tree to their immutable counterparts.

    This function traverses the input parameter tree and, for each element, checks if it is a mutable array.
    If so, it creates an immutable copy of the array; otherwise, the element is left unchanged.

    Args:
        vals (PyTree): The parameter tree potentially containing mutable arrays.

    Returns:
        PyTree: A new parameter tree where all mutable arrays have been replaced with immutable copies.
    """
    return jax.tree.map(lambda x: x[...] if _is_mutable_array(x) else x, vals)
