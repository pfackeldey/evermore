from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PyTree

import evermore as evm

jax.config.update("jax_enable_x64", True)


def test_minuit_transform_round_trip():
    transform = evm.transform.MinuitTransform()
    params: PyTree = {
        "bounded": evm.Parameter(
            value=jnp.array([0.25]),
            lower=jnp.array([0.0]),
            upper=jnp.array([1.0]),
            transform=transform,
        )
    }

    original_value = params["bounded"].value.copy()

    unconstrained = evm.transform.unwrap(params)
    assert jnp.allclose(params["bounded"].value, original_value)
    # value should now live in the unconstrained space
    assert jnp.all(unconstrained["bounded"].value < jnp.array([jnp.pi / 2]))

    constrained = evm.transform.wrap(unconstrained)
    assert jnp.allclose(constrained["bounded"].value, jnp.array([0.25]))


def test_minuit_transform_raises_for_out_of_bounds():
    transform = evm.transform.MinuitTransform()
    params: PyTree = {
        "bounded": evm.Parameter(
            value=jnp.array([1.5]),
            lower=jnp.array([0.0]),
            upper=jnp.array([1.0]),
            transform=transform,
        )
    }

    with pytest.raises(ValueError, match="value needs to be bounded between"):
        evm.transform.unwrap(params)


def test_softplus_transform_round_trip():
    transform = evm.transform.SoftPlusTransform()
    params: PyTree = {
        "positive": evm.Parameter(
            value=jnp.array([2.0]),
            transform=transform,
        )
    }

    original_value = params["positive"].value.copy()

    unconstrained = evm.transform.unwrap(params)
    assert jnp.allclose(params["positive"].value, original_value)
    # inverse softplus should place value on the real line (negative allowed)
    assert jnp.all(unconstrained["positive"].value < jnp.array([2.0]))

    reconstrained = evm.transform.wrap(unconstrained)
    assert jnp.allclose(reconstrained["positive"].value, jnp.array([2.0]))
