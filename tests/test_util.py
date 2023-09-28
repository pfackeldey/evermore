from __future__ import annotations

import jax

from dilax.util import FrozenDB, as1darray


def get_frozendb():
    return FrozenDB(
        {
            # QCD
            ("a", "b"): 1,
            ("a", "d", "e"): 2,
            ("a", "d", "f"): 3,
            # DY
            ("g", "b"): 4,
            ("g", "d", "e"): 5,
            ("g", "d", "f"): 6,
        }
    )


def test_frozendb_len():
    db = get_frozendb()

    assert len(db) == 6


def test_frozendb_getitem():
    db = get_frozendb()

    assert db["a"]["b"] == 1
    assert db["a", "b"] == 1
    assert db["b"] == FrozenDB({"a": 1, "g": 4})


def test_frozendb_jitcompatible():
    db = get_frozendb()

    @jax.jit
    def fun(db):
        return (db["a", "b"] + 1) ** 2

    assert fun(db) == 4


def test_as1darray():
    arr = as1darray(jax.numpy.array(1.0))

    assert isinstance(arr, jax.Array)
    assert arr.ndim == 1
