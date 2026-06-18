from __future__ import annotations

import typing as tp

import jax
import pytest
from jaxtyping import Float, Scalar

import evermore as evm
from evermore.pdf import Normal

jax.config.update("jax_enable_x64", True)

ScalarParam: tp.TypeAlias = evm.Parameter[Float[Scalar, ""]]


def test_Parameter():
    p: ScalarParam = evm.Parameter(value=1.0, lower=0.0, upper=2.0)
    assert p.get_value() == pytest.approx(1.0)
    assert p.lower == pytest.approx(0.0)
    assert p.upper == pytest.approx(2.0)
    assert p.prior is None


def test_NormalParameter():
    p: ScalarParam = evm.NormalParameter(value=1.0, lower=0.0, upper=2.0)
    assert p.get_value() == pytest.approx(1.0)
    assert p.lower == pytest.approx(0.0)
    assert p.upper == pytest.approx(2.0)
    assert isinstance(p.prior, Normal)
