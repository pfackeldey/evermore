from __future__ import annotations

import operator
from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree, Shaped

__all__ = [
    "_missing",
    "dump_hlo_graph",
    "dump_jaxpr",
    "filter_tree_map",
    "maybe_float_array",
    "sum_over_leaves",
    "tree_stack",
]


def __dir__():
    return __all__


def maybe_float_array(x: Any, passthrough: bool = True) -> Float[Array, "..."]:  # noqa: UP037
    if eqx.is_array_like(x):
        return jnp.asarray(x, jnp.result_type(float))
    if passthrough:
        return x
    msg = f"Expected an array-like object, got {type(x).__name__} instead."
    raise ValueError(msg)


@jax.tree_util.register_static
class _Missing:
    __slots__ = ()

    def __repr__(self):
        return "--"


_missing = _Missing()
del _Missing


def filter_tree_map(
    fun: Callable,
    tree: PyTree,
    filter: Callable,
) -> PyTree:
    filtered = eqx.filter(tree, filter, is_leaf=filter)
    return jax.tree.map(
        fun,
        filtered,
        is_leaf=filter,
    )


def sum_over_leaves(tree: PyTree) -> Array:
    return jax.tree.reduce(operator.add, tree)


def tree_stack(
    trees: list[PyTree[Shaped[Array, "..."]]],  # noqa: UP037
    *,
    broadcast_leaves: bool = False,
) -> PyTree[Shaped[Array, "batch_dim ..."]]:
    """
    Turns e.g. an array of evm.Modifier(s) into a evm.Modifier of arrays. (AOS -> SOA)
    The leaves can be broadcasted to the same shape if broadcast_effect is set to True.
    The stacked PyTree will have the static nodes of the first PyTree in the list.

    Example:

    .. code-block:: python

        import evermore as evm
        import jax
        import jax.numpy as jnp
        import wadler_lindig as wl

        modifiers = [
            evm.NormalParameter().scale_log(up=jnp.array([1.1]), down=jnp.array([0.9])),
            evm.NormalParameter().scale_log(up=jnp.array([1.2]), down=jnp.array([0.8])),
        ]
        wl.pprint(evm.util.tree_stack(modifiers), hide_defaults=False)
        # Modifier(
        #   parameter=NormalParameter(
        #     value=f32[2,1](jax),
        #     name=None,
        #     lower=None,
        #     upper=None,
        #     prior=Normal(mean=f32[2,1](jax), width=f32[2,1](jax)),
        #     frozen=False,
        #     transform=None
        #   ),
        #   effect=AsymmetricExponential(up=f32[2,1](jax), down=f32[2,1](jax))
        # )
    """
    dynamic_trees, static_trees = eqx.partition(trees, eqx.is_array)
    for tree in static_trees[1:]:
        if jax.tree.structure(tree) != jax.tree.structure(static_trees[0]):
            msg = (
                "All static trees must have the same structure. "
                f"Got {jax.tree.structure(tree)} and {jax.tree.structure(static_trees[0])}"
            )
            raise ValueError(msg)

    def batch_axis_stack(*leaves: Array) -> Array:
        leaves = jax.tree.map(jnp.atleast_1d, leaves)  # ensure at least 1D
        if broadcast_leaves:
            shape = jnp.broadcast_shapes(*[leaf.shape for leaf in leaves])
            return jnp.stack(
                jax.tree.map(partial(jnp.broadcast_to, shape=shape), leaves)
            )
        return jnp.stack(leaves, axis=0)

    dynamic_trees = jax.tree.map(batch_axis_stack, *dynamic_trees)
    return eqx.combine(static_trees[0], dynamic_trees)


def dump_jaxpr(fun: Callable, *args: Any, **kwargs: Any) -> str:
    """Helper function to dump the Jaxpr of a function.

    Example:

    .. code-block:: python

        import jax
        import jax.numpy as jnp


        def f(x: jax.Array) -> jax.Array:
            return jnp.sin(x) ** 2 + jnp.cos(x) ** 2


        x = jnp.array([1.0, 2.0, 3.0])

        print(dump_jaxpr(f, x))
        # -> { lambda ; a:f32[3]. let
        #        b:f32[3] = sin a              # []
        #        c:f32[3] = integer_pow[y=2] b # []
        #        d:f32[3] = cos a              # []
        #        e:f32[3] = integer_pow[y=2] d # []
        #        f:f32[3] = add c e            # []
        #      in (f,) }
    """
    jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
    return jaxpr.pretty_print(name_stack=True)


def dump_hlo_graph(fun: Callable, *args: Any, **kwargs: Any) -> str:
    """
    Helper to dump the HLO graph of a function as a `dot` graph.

    Example:

    .. code-block:: python

        import jax
        import jax.numpy as jnp

        import path


        def f(x: jax.Array) -> jax.Array:
            return x + 1.0


        x = jnp.array([1.0, 2.0, 3.0])

        # dump dot graph to file
        filepath = pathlib.Path("graph.gv")
        filepath.write_text(dump_hlo_graph(f, x), encoding="ascii")
    """
    return jax.jit(fun).lower(*args, **kwargs).compiler_ir("hlo").as_hlo_dot_graph()
