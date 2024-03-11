from __future__ import annotations

import jax

from evermore.util import as1darray


def test_as1darray():
    arr = as1darray(jax.numpy.array(1.0))

    assert isinstance(arr, jax.Array)
    assert arr.ndim == 1
