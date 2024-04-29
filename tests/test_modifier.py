from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import evermore as evm


def test_Modifier():
    param = evm.Parameter(value=1.1)
    modifier = param.scale()

    hist = jnp.array([1, 2, 3])

    assert isinstance(modifier, evm.Modifier)
    np.testing.assert_allclose(modifier(hist), jnp.array([1.1, 2.2, 3.3]))


def test_Where():
    param1 = evm.Parameter(value=1.0)
    param2 = evm.Parameter(value=1.1)
    modifier1 = param1.scale()
    modifier2 = param2.scale()

    hist = jnp.array([1, 2, 3])

    where_mod = evm.modifier.Where(hist > 1.5, modifier2, modifier1)
    np.testing.assert_allclose(where_mod(hist), jnp.array([1, 2.2, 3.3]))


def test_BooleanMask():
    param = evm.Parameter(value=1.1)
    modifier = param.scale()

    hist = jnp.array([1, 2, 3])

    masked_mod = evm.modifier.BooleanMask(jnp.array([True, False, True]), modifier)
    np.testing.assert_allclose(masked_mod(hist), jnp.array([1.1, 2, 3.3]))


def test_Transform():
    param = evm.Parameter(value=1.1)
    modifier = param.scale()

    hist = jnp.array([1, 2, 3])

    sqrt_modifier = evm.modifier.Transform(jnp.sqrt, modifier)
    np.testing.assert_allclose(
        sqrt_modifier(hist), jnp.array([1.0488088, 2.0976176, 3.1464264])
    )


def test_mix_modifiers():
    param = evm.Parameter(value=1.1)
    modifier = param.scale()

    hist = jnp.array([1, 2, 3])

    sqrt_modifier = evm.modifier.Transform(jnp.sqrt, modifier)
    sqrt_masked_modifier = evm.modifier.BooleanMask(
        jnp.array([True, False, True]), sqrt_modifier
    )
    np.testing.assert_allclose(
        sqrt_masked_modifier(hist), jnp.array([1.0488088, 2, 3.1464264])
    )


def test_Compose():
    param1 = evm.Parameter(value=1.0)
    param2 = evm.Parameter(value=1.1)
    modifier1 = param1.scale()
    modifier2 = param2.scale()

    hist = jnp.array([1, 2, 3])

    composition = modifier1 @ modifier2
    np.testing.assert_allclose(composition(hist), jnp.array([1.1, 2.2, 3.3]))
