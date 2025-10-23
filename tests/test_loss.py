from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Float, PyTree, Scalar

import evermore as evm

jax.config.update("jax_enable_x64", True)


ScalarParam: tp.TypeAlias = evm.Parameter[Float[Scalar, ""]]
ScalarParamTree: tp.TypeAlias = PyTree[ScalarParam]


def loss_fn(params: ScalarParamTree) -> Float[Scalar, ""]:
    return (
        params["a"].value ** 2
        + 2 * params["b"].value ** 2
        + (params["a"].value + params["c"].value) ** 2
    )


params: ScalarParamTree = {
    "a": evm.Parameter(2.0),
    "b": evm.Parameter(3.0),
    "c": evm.Parameter(4.0),
}


def test_get_log_probs():
    # use some with constraints
    params: ScalarParamTree = {
        "a": evm.NormalParameter(value=0.5),
        "b": evm.NormalParameter(),
        "c": evm.Parameter(),
    }

    log_probs = evm.loss.get_log_probs(params)
    assert log_probs["a"] == pytest.approx(-0.125)
    assert log_probs["b"] == pytest.approx(0.0)
    assert log_probs["c"] == pytest.approx(0.0)


def test_covariance_matrix():
    cov = evm.loss.covariance_matrix(loss_fn, params)

    assert cov.shape == (3, 3)
    expected = jnp.array(
        [[1.0, 0.0, -0.70710677], [0.0, 1.0, 0.0], [-0.70710677, 0.0, 1.0]]
    )
    np.testing.assert_allclose(cov, expected, rtol=1e-6, atol=1e-8)


def test_hessian_matrix():
    h = evm.loss.hessian_matrix(loss_fn, params)

    assert h.shape == (3, 3)
    np.testing.assert_allclose(
        h,
        jnp.array([[4.0, 0.0, 2.0], [0.0, 4.0, 0.0], [2.0, 0.0, 2.0]]),
    )


def test_cramer_rao_uncertainty():
    uncertainty = evm.loss.cramer_rao_uncertainty(loss_fn, params)

    assert uncertainty["a"] == pytest.approx(0.70710677)
    assert uncertainty["b"] == pytest.approx(0.5)
    assert uncertainty["c"] == pytest.approx(1.0)
