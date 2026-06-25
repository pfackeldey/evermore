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
    params = {
        "a": evm.Parameter(value=jnp.array([1.0])),
        "b": evm.Parameter(value=jnp.array([2.0])),
    }

    rngs = nnx.Rngs(0)
    cov = jnp.eye(2)

    sampled = evm.sample.sample_from_covariance_matrix(
        rngs,
        params,
        covariance_matrix=cov,
        n_samples=5,
    )

    assert sampled["a"].get_value().shape == (5, 1)
    assert sampled["b"].get_value().shape == (5, 1)
    # original parameters remain unchanged
    assert np.allclose(params["a"].get_value(), jnp.array([1.0]))
    assert np.allclose(params["b"].get_value(), jnp.array([2.0]))

    # deterministic across identical seeds
    rngs_again = nnx.Rngs(0)
    sampled_again = evm.sample.sample_from_covariance_matrix(
        rngs_again,
        params,
        covariance_matrix=cov,
        n_samples=5,
    )
    np.testing.assert_allclose(
        sampled["a"].get_value(),
        sampled_again["a"].get_value(),
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

    assert sampled["plain"].get_value() == pytest.approx(3.0)
    assert sampled["normal"].get_value().shape == (1,)
    assert params["normal"].get_value() == pytest.approx(0.0)

    # sampling with the same seed should be reproducible
    rngs_again = nnx.Rngs(123)
    sampled_again = evm.sample.sample_from_priors(rngs_again, params)
    np.testing.assert_allclose(
        sampled["normal"].get_value(),
        sampled_again["normal"].get_value(),
    )


class _PoissonDiscreteParameter(evm.Parameter):
    @property
    def prior(self):
        return evm.pdf.PoissonDiscrete(lamb=jnp.array([5.0]))


class _PoissonContinuousParameter(evm.Parameter):
    @property
    def prior(self):
        return evm.pdf.PoissonContinuous(lamb=jnp.array([5.0]))


def test_sample_from_priors_poisson_discrete():
    param = _PoissonDiscreteParameter(value=jnp.array([0.0]))
    rngs = nnx.Rngs(0)
    sampled = evm.sample.sample_from_priors(rngs, {"p": param})
    assert sampled["p"].get_value().shape == (1,)
    assert np.isfinite(sampled["p"].get_value()).all()
    # original unchanged
    assert np.allclose(param.get_value(), jnp.array([0.0]))


def test_sample_from_priors_poisson_continuous_raises():
    param = _PoissonContinuousParameter(value=jnp.array([0.0]))
    rngs = nnx.Rngs(0)
    with pytest.raises(NotImplementedError):
        evm.sample.sample_from_priors(rngs, {"p": param})


def test_sample_from_covariance_matrix_mask():
    params = {
        "a": evm.Parameter(value=jnp.array([1.0])),
        "b": evm.Parameter(value=jnp.array([2.0])),
    }

    rngs = nnx.Rngs(0)
    cov = jnp.eye(2)

    # mask out parameter "a", only "b" should be resampled
    mask = {"a": False, "b": True}

    sampled = evm.sample.sample_from_covariance_matrix(
        rngs,
        params,
        covariance_matrix=cov,
        mask=mask,
        n_samples=5,
    )

    # "a" should keep its original value for all samples
    assert sampled["a"].get_value().shape == (5, 1)
    assert np.allclose(sampled["a"].get_value(), 1.0)
    # "b" should be resampled (different from original 2.0)
    assert sampled["b"].get_value().shape == (5, 1)
    assert not np.allclose(sampled["b"].get_value(), 2.0)
