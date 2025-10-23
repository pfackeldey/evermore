from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jaxtyping import PyTree

import evermore as evm

jax.config.update("jax_enable_x64", True)


def test_sample_from_covariance_matrix_preserves_structure():
    params: PyTree = {
        "a": evm.Parameter(value=jnp.array([1.0])),
        "b": evm.Parameter(value=jnp.array([2.0])),
        "meta": 42.0,
    }

    rngs = nnx.Rngs(0)
    cov = jnp.eye(2)

    sampled: PyTree[evm.Parameter] = evm.sample.sample_from_covariance_matrix(
        rngs,
        params,
        covariance_matrix=cov,
        n_samples=5,
    )

    assert sampled["meta"] == 42.0
    assert sampled["a"].value.shape == (5, 1)
    assert sampled["b"].value.shape == (5, 1)
    # original parameters remain unchanged
    assert np.allclose(params["a"].value, jnp.array([1.0]))
    assert np.allclose(params["b"].value, jnp.array([2.0]))

    # deterministic across identical seeds
    rngs_again = nnx.Rngs(0)
    sampled_again = evm.sample.sample_from_covariance_matrix(
        rngs_again,
        params,
        covariance_matrix=cov,
        n_samples=5,
    )
    np.testing.assert_allclose(
        sampled["a"].value,
        sampled_again["a"].value,
    )


def test_sample_from_priors_respects_priors_and_frozen_parameters():
    normal_prior = evm.pdf.Normal(mean=jnp.array([0.0]), width=jnp.array([1.0]))
    params: PyTree[evm.Parameter] = {
        "normal": evm.Parameter(
            value=jnp.array([0.0]),
            prior=normal_prior,
        ),
        "plain": evm.Parameter(value=jnp.array([3.0])),
    }

    rngs = nnx.Rngs(123)
    sampled: PyTree[evm.Parameter] = evm.sample.sample_from_priors(rngs, params)

    assert sampled["plain"].value == pytest.approx(3.0)
    assert sampled["normal"].value.shape == (1,)
    assert params["normal"].value == pytest.approx(0.0)

    # sampling with the same seed should be reproducible
    rngs_again = nnx.Rngs(123)
    sampled_again = evm.sample.sample_from_priors(rngs_again, params)
    np.testing.assert_allclose(
        sampled["normal"].value,
        sampled_again["normal"].value,
    )
