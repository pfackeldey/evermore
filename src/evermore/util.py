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
