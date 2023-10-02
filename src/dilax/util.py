from __future__ import annotations

import collections
import pprint
from collections.abc import Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import jax
import jax.numpy as jnp


class Sentinel:
    __slots__ = ("repr",)

    def __init__(self, repr: str) -> None:
        self.repr = repr

    def __repr__(self) -> str:
        return self.repr

    __str__ = __repr__


_NoValue: Sentinel = Sentinel("<NoValue>")


class FrozenKeysView(collections.abc.KeysView):
    """FrozenKeysView that does not print values when repr'ing."""

    def __init__(self, mapping):
        super().__init__(mapping)
        self._mapping = mapping

    def __repr__(self):
        return f"{type(self).__name__}({list(map(_pretty_key, self._mapping.keys()))})"

    __str__ = __repr__


def _pretty_key(key):
    if not isinstance(key, frozenset):
        key = FrozenDB.keyify(key)
    if len(key) == 1:
        return next(iter(key))
    return tuple([_pretty_key(k) for k in key])


def _indent(amount: int, s: str) -> str:
    """Indents `s` with `amount` spaces."""
    prefix = amount * " "
    return "\n".join(prefix + line for line in s.splitlines())


def _pretty_dict(x):
    if not isinstance(x, Mapping):
        return pprint.pformat(x)
    rep = ""
    for key, val in x.items():
        rep += f"{_pretty_key(key)!r}: {_pretty_dict(val)},\n"
    if rep:
        return "{\n" + _indent(2, rep) + "\n}"
    return "{}"


K = TypeVar("K")
V = TypeVar("V")


def _prepare_freeze(xs: Any) -> Any:
    """Deep copy unfrozen dicts to make the dictionary FrozenDict safe."""
    if isinstance(xs, FrozenDB):
        # we can safely ref share the internal state of a FrozenDict
        # because it is immutable.
        return xs._dict
    if not isinstance(xs, dict):
        # return a leaf as is.
        return xs
    # recursively copy dictionary to avoid ref sharing
    return {FrozenDB.keyify(key): _prepare_freeze(val) for key, val in xs.items()}


def _check_no_duplicate_keys(keys: Iterable[Hashable]) -> None:
    keys = list(keys)
    if any(keys.count(x) > 1 for x in keys):
        msg = f"Duplicate keys: {tuple(keys)}, this is not allowed!"
        raise ValueError(msg)


class FrozenDB(Mapping[K, V]):
    """An immutable database-like custom dict.

    Example:

    .. code-block:: python

        hists = HistDB(
            {
                # QCD
                ("QCD", "nominal"): jnp.array([1, 1, 1, 1, 1]),
                ("QCD", "JES", "Up"): jnp.array([1.5, 1.5, 1.5, 1.5, 1.5]),
                ("QCD", "JES", "Down"): jnp.array([0.5, 0.5, 0.5, 0.5, 0.5]),
                # DY
                ("DY", "nominal"): jnp.array([2, 2, 2, 2, 2]),
                ("DY", "JES", "Up"): jnp.array([2.5, 2.5, 2.5, 2.5, 2.5]),
                ("DY", "JES", "Down"): jnp.array([0.7, 0.7, 0.7, 0.7, 0.7]),
            }
        )

        print(hists)
        # -> HistDB({
        #   ('QCD', 'nominal'): Array([1, 1, 1, 1, 1], dtype=int32),
        #   ('QCD', 'Up', 'JES'): Array([1.5, 1.5, 1.5, 1.5, 1.5], dtype=float32),
        #   ('QCD', 'Down', 'JES'): Array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float32),
        #   ('DY', 'nominal'): Array([2, 2, 2, 2, 2], dtype=int32),
        #   ('DY', 'Up', 'JES'): Array([2.5, 2.5, 2.5, 2.5, 2.5], dtype=float32),
        #   ('DY', 'Down', 'JES'): Array([0.7, 0.7, 0.7, 0.7, 0.7], dtype=float32),
        # })

        print(hists["QCD"])
        # -> HistDB({
        #     'nominal': Array([1, 1, 1, 1, 1], dtype=int32),
        #     ('Up', 'JES'): Array([1.5, 1.5, 1.5, 1.5, 1.5], dtype=float32),
        #     ('Down', 'JES'): Array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float32),
        # })

        print(hists["JES"])
        # -> HistDB({
        #     ('QCD', 'Up'): Array([1.5, 1.5, 1.5, 1.5, 1.5], dtype=float32),
        #     ('QCD', 'Down'): Array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float32),
        #     ('DY', 'Up'): Array([2.5, 2.5, 2.5, 2.5, 2.5], dtype=float32),
        #     ('DY', 'Down'): Array([0.7, 0.7, 0.7, 0.7, 0.7], dtype=float32),
        # })

        # It's jit-compatible:
        def foo(hists):
            return (hists["QCD", "nominal"] + 1.2) ** 2

        print(jax.jit(foo)(hists))
        # -> Array([4.84, 4.84, 4.84, 4.84, 4.84], dtype=float32, weak_type=True)
    """

    __slots__ = ("_dict",)

    if TYPE_CHECKING:
        _dict: dict[frozenset, Any]

    @staticmethod
    def keyify(keyish: Any) -> frozenset:
        if not isinstance(keyish, (tuple, list, set, frozenset)):
            keyish = (keyish,)
        _check_no_duplicate_keys(keyish)
        keyish = frozenset(keyish)
        assert not any(isinstance(key, set) for key in keyish)
        return keyish

    def __init__(
        self,
        xs: Mapping | Sentinel = _NoValue,
        __unsafe_skip_copy__: bool = False,
    ) -> None:
        # make sure the dict is as
        if xs is _NoValue:
            xs = {}
        data = dict(cast(Mapping, xs))
        if __unsafe_skip_copy__:
            self._dict = data
        else:
            self._dict = _prepare_freeze(data)

    def __getitem__(self, key) -> Any:
        key = self.keyify(key)
        if key in self._dict:
            return self._dict[key]
        ret = self.__class__({k - key: v for k, v in self.items() if key <= k})
        if not ret:
            raise KeyError(key)
        return ret

    def __setitem__(self, key, value) -> None:
        msg = f"{type(self).__name__} is immutable."
        raise ValueError(msg)

    def __contains__(self, key) -> bool:
        key = self.keyify(key)
        return key in self._dict

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def keys(self) -> FrozenKeysView:
        return FrozenKeysView(self._dict)

    def values(self):
        return self._dict.values()

    def items(self):
        for key in self._dict:
            yield (key, self[key])

    def only(self, *keys) -> FrozenDB:
        return self.__class__({key: self[key] for key in keys})

    def subset(self, *keys) -> FrozenDB:
        new = {}
        for key in keys:
            new.update({k: v for k, v in self.items() if self.keyify(key) <= k})
        return self.__class__(new)

    def copy(self) -> FrozenDB:
        return self.__class__(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({_pretty_dict(self._dict)})"

    def as_compact_dict(self):
        return {"/".join(sorted(map(str, k))): v for k, v in self.items()}


def _flatten(tree):
    return (tuple(tree.values()), tuple(tree.keys()))


def _make_unflatten(cls: type[FrozenDB]) -> Callable:
    def _unflatten(keys, values):
        return cls(dict(zip(keys, values)), __unsafe_skip_copy__=True)

    return _unflatten


class HistDB(FrozenDB):
    ...


# then we register them with jax as a PyTree
for cls in HistDB, FrozenDB:
    jax.tree_util.register_pytree_node(
        cls,
        _flatten,
        _make_unflatten(cls),
    )


def as1darray(x: float | jax.Array) -> jax.Array:
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
