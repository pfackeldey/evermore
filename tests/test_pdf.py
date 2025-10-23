from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evermore.pdf import (
    Normal,
    PoissonContinuous,
    PoissonDiscrete,
    discrete_inv_cdf_search,
)

jax.config.update("jax_enable_x64", True)


def test_Normal():
    pdf = Normal(mean=jnp.array(0.0), width=jnp.array(1.0))

    assert pdf.log_prob(jnp.array(0.0)) == pytest.approx(0.0)


def test_PoissonDiscrete():
    pdf = PoissonDiscrete(lamb=jnp.array(10))

    assert pdf.log_prob(jnp.array(5.0)) == pytest.approx(-1.5342636)


def test_PoissonContinuous():
    pdf = PoissonContinuous(lamb=jnp.array(10))

    assert pdf.log_prob(jnp.array(5.0)) == pytest.approx(-1.5342636)


def test_discrete_inv_cdf_search():
    lamb = 5.0

    def start_fn(x):
        return jnp.floor(lamb + jax.scipy.stats.norm.ppf(x) * jnp.sqrt(lamb))

    def cdf_fn(k):
        return jax.scipy.stats.poisson.cdf(k, lamb)

    # test correct algorithmic behavior
    assert discrete_inv_cdf_search(jnp.array([0.9]), cdf_fn, start_fn, "floor") == 7
    assert discrete_inv_cdf_search(jnp.array([0.9]), cdf_fn, start_fn, "ceil") == 8
    assert discrete_inv_cdf_search(jnp.array([0.9]), cdf_fn, start_fn, "closest") == 8

    # test individual solutions in vmapped mode plus shape preservation
    k = discrete_inv_cdf_search(jnp.array([0.9, 0.95, 0.99]), cdf_fn, start_fn, "floor")
    np.testing.assert_allclose(k, jnp.array([7.0, 8.0, 10.0]))
    k = discrete_inv_cdf_search(jnp.array([0.9, 0.95, 0.99]), cdf_fn, start_fn, "ceil")
    np.testing.assert_allclose(k, jnp.array([8.0, 9.0, 11.0]))
    k = discrete_inv_cdf_search(
        jnp.array([0.9, 0.95, 0.99]), cdf_fn, start_fn, "closest"
    )
    np.testing.assert_allclose(k, jnp.array([8.0, 8.0, 10.0]))
