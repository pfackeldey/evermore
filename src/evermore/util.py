from __future__ import annotations

import operator
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree, Shaped

__all__ = [
    "dump_hlo_graph",
    "dump_jaxpr",
    "float_array",
    "sum_over_leaves",
    "tree_stack",
]


def __dir__():
    return __all__


def float_array(x: Any) -> Float[Array, "..."]:  # noqa: UP037
    return jnp.asarray(x, dtype=jnp.result_type(jnp.float_))


def sum_over_leaves(tree: PyTree) -> Array:
    return jax.tree.reduce_associative(operator.add, tree)


def tree_stack(
    trees: list[PyTree[Shaped[Array, "..."]]],  # noqa: UP037
    *,
    broadcast_leaves: bool = False,
) -> PyTree[Shaped[Array, "batch_dim ..."]]:
    """Stacks a list of PyTrees into a batched PyTree (AOS → SOA).

    Args:
        trees: Sequence of PyTrees with identical static structure.
        broadcast_leaves: Whether to broadcast each leaf to a common shape before stacking.

    Returns:
        PyTree[Shaped[Array, "batch_dim ..."]]: PyTree whose leaves now carry an
            additional leading batch dimension.

    Examples:
        >>> import evermore as evm
        >>> import jax.numpy as jnp
        >>> modifiers = [
        ...     evm.NormalParameter().scale_log_asymmetric(up=jnp.array([1.1]), down=jnp.array([0.9])),
        ...     evm.NormalParameter().scale_log_asymmetric(up=jnp.array([1.2]), down=jnp.array([0.8])),
        ... ]
        >>> stacked = evm.util.tree_stack(modifiers)
        >>> stacked.parameter.value.shape
        (2, 1)
    """
    # check that all trees have the same structure
    first_treedef = jax.tree.structure(trees[0])
    for tree in trees[1:]:
        other_treedef = jax.tree.structure(tree)
        if other_treedef != first_treedef:
            msg = (
                "All static trees must have the same structure. "
                f"Got {other_treedef} and {first_treedef}"
            )
            raise ValueError(msg)

    # actual stacking function for leaves
    def batch_axis_stack(*leaves: Array) -> Array:
        leaves = jax.tree.map(jnp.atleast_1d, leaves)  # ensure at least 1D
        if broadcast_leaves:
            shape = jnp.broadcast_shapes(*(leaf.shape for leaf in leaves))
            return jnp.stack(
                jax.tree.map(partial(jnp.broadcast_to, shape=shape), leaves)
            )
        return jnp.stack(leaves, axis=0)

    return jax.tree.map(batch_axis_stack, *trees)


def dump_jaxpr(fun: Callable, *args: Any, **kwargs: Any) -> str:
    """Pretty-prints the Jaxpr of ``fun`` evaluated at the given arguments.

    Args:
        fun: Callable to analyse.
        *args: Positional arguments passed to ``fun``.
        **kwargs: Keyword arguments forwarded to ``fun``.

    Returns:
        str: Human-readable representation of the traced Jaxpr.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> def f(x):
        ...     return jnp.sin(x) ** 2 + jnp.cos(x) ** 2
        >>> print(dump_jaxpr(f, jnp.array([1.0, 2.0, 3.0])))
        { lambda ; a:f32[3]. let ... }
    """
    jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
    return jaxpr.pretty_print(name_stack=True)


def dump_hlo_graph(fun: Callable, *args: Any, **kwargs: Any) -> str:
    """Returns the HLO ``dot`` graph of ``fun`` evaluated at the inputs.

    Args:
        fun: Callable to trace.
        *args: Positional arguments passed to ``fun``.
        **kwargs: Keyword arguments forwarded to ``fun``.

    Returns:
        str: ``dot`` graph describing the lowered HLO program.

    Examples:
        >>> import pathlib
        >>> import jax.numpy as jnp
        >>> def f(x):
        ...     return x + 1.0
        >>> graph = dump_hlo_graph(f, jnp.array([1.0, 2.0, 3.0]))
        >>> pathlib.Path("graph.gv").write_text(graph, encoding="ascii")
        143
    """
    return jax.jit(fun).lower(*args, **kwargs).compiler_ir("hlo").as_hlo_dot_graph()  # ty:ignore[possibly-missing-attribute]
