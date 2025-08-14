from __future__ import annotations

import typing as tp

from jaxtyping import Float, Scalar

import evermore as evm
from evermore.pdf import Normal

ScalarParam: tp.TypeAlias = evm.Parameter[Float[Scalar, ""]]


def test_Parameter():
    p: ScalarParam = evm.Parameter(value=1.0, lower=0.0, upper=2.0)
    assert p.value == 1.0
    assert p.lower == 0.0
    assert p.upper == 2.0
    assert p.prior is None


def test_NormalParameter():
    p: ScalarParam = evm.NormalParameter(value=1.0, lower=0.0, upper=2.0)
    assert p.value == 1.0
    assert p.lower == 0.0
    assert p.upper == 2.0
    assert isinstance(p.prior, Normal)
