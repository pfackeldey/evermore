from __future__ import annotations

import typing as tp

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, PyTree, Scalar

import evermore as evm

ScalarParam: tp.TypeAlias = evm.Parameter[Float[Scalar, ""]]
ScalarParamTree: tp.TypeAlias = PyTree[ScalarParam]


def test_Modifier():
    param: ScalarParam = evm.Parameter(value=1.1)
    modifier = param.scale()

    hist = jnp.array([1, 2, 3])

    assert isinstance(modifier, evm.Modifier)
    np.testing.assert_allclose(modifier(hist), jnp.array([1.1, 2.2, 3.3]))


def test_Where():
    param1: ScalarParam = evm.Parameter(value=1.0)
    param2: ScalarParam = evm.Parameter(value=1.1)
    modifier1 = param1.scale()
    modifier2 = param2.scale()

    hist = jnp.array([1, 2, 3])

    where_mod: evm.modifier.Where[ScalarParamTree] = evm.modifier.Where(
        hist > 1.5, modifier2, modifier1
    )
    np.testing.assert_allclose(where_mod(hist), jnp.array([1, 2.2, 3.3]))


def test_BooleanMask():
    param: ScalarParam = evm.Parameter(value=1.1)
    modifier = param.scale()

    hist = jnp.array([1, 2, 3])

    masked_mod: evm.modifier.BooleanMask[ScalarParamTree] = evm.modifier.BooleanMask(
        jnp.array([True, False, True]), modifier
    )
    np.testing.assert_allclose(masked_mod(hist), jnp.array([1.1, 2, 3.3]))


def test_Transform():
    param: ScalarParam = evm.Parameter(value=1.1)
    modifier = param.scale()

    hist = jnp.array([1, 2, 3])

    sqrt_modifier: evm.modifier.Transform[ScalarParamTree] = evm.modifier.Transform(
        jnp.sqrt, modifier
    )
    np.testing.assert_allclose(
        sqrt_modifier(hist), jnp.array([1.0488088, 2.0976176, 3.1464264])
    )


def test_mix_modifiers():
    param: ScalarParam = evm.Parameter(value=1.1)
    modifier = param.scale()

    hist = jnp.array([1, 2, 3])

    sqrt_modifier: evm.modifier.Transform[ScalarParamTree] = evm.modifier.Transform(
        jnp.sqrt, modifier
    )
    sqrt_masked_modifier: evm.modifier.BooleanMask[ScalarParamTree] = (
        evm.modifier.BooleanMask(jnp.array([True, False, True]), sqrt_modifier)
    )
    np.testing.assert_allclose(
        sqrt_masked_modifier(hist), jnp.array([1.0488088, 2, 3.1464264])
    )


def test_Compose():
    param1: ScalarParam = evm.Parameter(value=1.0)
    param2: ScalarParam = evm.Parameter(value=1.1)
    modifier1 = param1.scale()
    modifier2 = param2.scale()

    hist = jnp.array([1, 2, 3])

    composition = modifier1 @ modifier2
    np.testing.assert_allclose(composition(hist), jnp.array([1.1, 2.2, 3.3]))
