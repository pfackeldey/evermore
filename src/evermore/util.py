from __future__ import annotations

import operator
from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, PyTree

__all__ = [
    "is_parameter",
    "sum_leaves",
    "tree_stack",
    "as1darray",
    "dump_hlo_graph",
    "dump_jaxpr",
]


def __dir__():
    return __all__


def is_parameter(leaf: Any) -> bool:
    from evermore import Parameter

    return isinstance(leaf, Parameter)


def _filtered_module_map(
    module: eqx.Module,
    fun: Callable,
    filter: Callable,
) -> eqx.Module:
    params = eqx.filter(module, filter, is_leaf=filter)
    return jtu.tree_map(
        fun,
        params,
        is_leaf=filter,
    )


_params_map = partial(_filtered_module_map, filter=is_parameter)


def sum_leaves(tree: PyTree) -> Array:
    return jtu.tree_reduce(operator.add, tree)


def tree_stack(trees: list[PyTree], broadcast_leaves: bool = False) -> PyTree:
    """
    Turn an array of `evm.Modifier`(s) into a `evm.Modifier` of arrays.
    Caution:
        It is important that the `jax.Array`(s) of the underlying `evm.Parameter` have the same shape.
        Same applies for the effect leaves (e.g. `width`). However, the effect leaves can be
        broadcasted to the same shape if `broadcast_effect_leaves` is set to `True`.

    Example:

        .. code-block:: python

            import evermore as evm
            import jax
            import jax.numpy as jnp

            modifier = [
                evm.Parameter().lnN(up=jnp.array([0.9, 0.95]), down=jnp.array([1.1, 1.14])),
                evm.Parameter().lnN(up=jnp.array([0.8]), down=jnp.array([1.2])),
            ]
            print(modifier_stack2(modifier))
            # -> Modifier(
            #      parameter=Parameter(
            #        value=f32[2,1], # <- stacked dimension (2, 1)
            #        lower=f32[1],
            #        upper=f32[1],
            #        constraint=Gauss(mean=f32[1], width=f32[1])
            #      ),
            #      effect=lnN(up=f32[2,1], down=f32[2,1]) # <- stacked dimension (2, 1)
            #    )
    """
    # If there is only one modifier, we can return it directly
    if len(trees) == 1:
        return trees[0]
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jtu.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list, strict=False)
    stacked_leaves = []
    for leaves in grouped_leaves:  # type: ignore[assignment]
        if broadcast_leaves:
            shape = jnp.broadcast_shapes(*[leaf.shape for leaf in leaves])
            stacked_leaves.append(
                jnp.stack(jtu.tree_map(partial(jnp.broadcast_to, shape=shape), leaves))
            )
        else:
            stacked_leaves.append(jnp.stack(leaves))
    return jtu.tree_unflatten(treedef_list[0], stacked_leaves)


def as1darray(x: ArrayLike) -> Array:
    """
    Converts `x` to a 1d array.

    Example:

    .. code-block:: python

        import jax.numpy as jnp


        as1darray(1.0)
        # -> Array([1.], dtype=float32, weak_type=True)

        as1darray(jnp.array(1.0))
        # -> Array([1.], dtype=float32, weak_type=True)
    """

    return jnp.atleast_1d(jnp.asarray(x))


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
        filepath = pathlib.Path('graph.gv')
        filepath.write_text(dump_hlo_graph(f, x), encoding='ascii')
    """
    return jax.xla_computation(fun)(*args, **kwargs).as_hlo_dot_graph()
