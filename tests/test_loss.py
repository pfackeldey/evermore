from __future__ import annotations

import typing as tp

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Float, PyTree, Scalar

import evermore as evm

ScalarParam: tp.TypeAlias = evm.Parameter[Float[Scalar, ""]]
ScalarParamTree: tp.TypeAlias = PyTree[ScalarParam]


def test_get_log_probs():
    params: ScalarParamTree = {
        "a": evm.NormalParameter(value=0.5),
        "b": evm.NormalParameter(),
        "c": evm.Parameter(),
    }

    log_probs = evm.loss.get_log_probs(params)
    assert log_probs["a"] == pytest.approx(-0.125)
    assert log_probs["b"] == pytest.approx(0.0)
    assert log_probs["c"] == pytest.approx(0.0)


def test_compute_covariance():
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

    cov = evm.loss.compute_covariance(loss_fn, params)

    assert cov.shape == (3, 3)
    np.testing.assert_allclose(
        cov,
        jnp.array([[1.0, 0.0, -0.7071067], [0.0, 1.0, 0.0], [-0.7071067, 0.0, 1.0]]),
    )
