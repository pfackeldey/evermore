from __future__ import annotations

import operator
from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

__all__ = [
    "atleast_1d_float_array",
    "dump_hlo_graph",
    "dump_jaxpr",
    "filter_tree_map",
    "sum_over_leaves",
    "tree_stack",
]


def __dir__():
    return __all__


def atleast_1d_float_array(x: Any) -> Array:
    float_type = jnp.result_type(float)
    return jnp.atleast_1d(jnp.asarray(x, dtype=float_type))


def filter_tree_map(
    fun: Callable,
    module: PyTree,
    filter: Callable,
) -> PyTree:
    params = eqx.filter(module, filter, is_leaf=filter)
    return jax.tree.map(
        fun,
        params,
        is_leaf=filter,
    )


def sum_over_leaves(tree: PyTree) -> Array:
    return jax.tree.reduce(operator.add, tree)


def tree_stack(trees: list[PyTree], broadcast_leaves: bool = False) -> PyTree:
    """
    Turns e.g. an array of evm.Modifier(s) into a evm.Modifier of arrays.

    It is important that the jax.Array(s) of the underlying Arrays have the same shape.
    Same applies for the effect leaves (e.g. width). However, the effect leaves can be
    broadcasted to the same shape if broadcast_effect_leaves is set to True.

    The stacked PyTree will have the static nodes of the first PyTree in the list.

    Example:

    .. code-block:: python

        import evermore as evm
        import jax
        import jax.numpy as jnp

        modifiers = [
            evm.NormalParameter().scale_log(up=jnp.array([1.1]), down=jnp.array([0.9])),
            evm.NormalParameter().scale_log(up=jnp.array([1.2]), down=jnp.array([0.8])),
        ]
        print(evm.util.tree_stack(modifiers))
        # -> Modifier(
        #      parameter=NormalParameter(
        #        name=None,
        #        value=f32[2,1], # <- stacked dimension (2, 1)
        #        lower=f32[2,1], # <- stacked dimension (2, 1)
        #        upper=f32[2,1], # <- stacked dimension (2, 1)
        #        prior=Normal(mean=f32[2,1], width=f32[2,1]), # <- stacked dimension (2,1)
        #        frozen=False
        #      ),
        #      effect=AsymmetricExponential(up=f32[2,1], down=f32[2,1]) # <- stacked dimension (2, 1)
        #    )
    """
    # If there is only one modifier, we can return it directly
    if len(trees) == 1:
        return trees[0]
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree.flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list, strict=False)
    stacked_leaves = []
    for leaves in grouped_leaves:  # type: ignore[assignment]
        if broadcast_leaves:
            shape = jnp.broadcast_shapes(*[leaf.shape for leaf in leaves])
            stacked_leaves.append(
                jnp.stack(jax.tree.map(partial(jnp.broadcast_to, shape=shape), leaves))
            )
        else:
            stacked_leaves.append(jnp.stack(leaves))
    return jax.tree.unflatten(treedef_list[0], stacked_leaves)


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
    return jax.jit(fun).lower(*args, **kwargs).compiler_ir("hlo").as_hlo_dot_graph()  # type: ignore[union-attr]
