from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import evermore as evm


def test_get_log_probs():
    params = {
        "a": evm.NormalParameter(value=0.5),  # type: ignore[arg-type]
        "b": evm.NormalParameter(),
        "c": evm.Parameter(),
    }

    log_probs = evm.loss.get_log_probs(params)
    assert log_probs["a"] == pytest.approx(-0.125)
    assert log_probs["b"] == pytest.approx(0.0)
    assert log_probs["c"] == pytest.approx(0.0)


def test_compute_covariance():
    def loss_fn(params):
        return (
            params["a"].value ** 2
            + 2 * params["b"].value ** 2
            + (params["a"].value + params["c"].value) ** 2
        )

    params = {"a": evm.Parameter(2.0), "b": evm.Parameter(3.0), "c": evm.Parameter(4.0)}

    cov = evm.loss.compute_covariance(loss_fn, params)

    assert cov.shape == (3, 3)
    np.testing.assert_allclose(
        cov,
        jnp.array([[1.0, 0.0, -0.7071067], [0.0, 1.0, 0.0], [-0.7071067, 0.0, 1.0]]),
    )
