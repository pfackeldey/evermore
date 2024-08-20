from __future__ import annotations

import jax.numpy as jnp
import pytest

import evermore as evm


def test_get_log_probs():
    params = {
        "a": evm.NormalParameter(value=0.5),
        "b": evm.NormalParameter(),
        "c": evm.Parameter(),
    }

    log_probs = evm.loss.get_log_probs(params)
    assert log_probs["a"] == pytest.approx(-0.125)
    assert log_probs["b"] == pytest.approx(0.0)
    assert log_probs["c"] == pytest.approx(0.0)


def test_get_boundary_constraints():
    in_bounds_param = evm.Parameter(value=1.0, lower=0.0, upper=2.0)
    oob_param = evm.Parameter(value=3.0, lower=0.0, upper=2.0)

    assert evm.loss.get_boundary_constraints(in_bounds_param) == 0.0
    assert evm.loss.get_boundary_constraints(oob_param) == jnp.inf
