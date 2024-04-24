from __future__ import annotations

import evermore as evm
from evermore.pdf import Normal


def test_Parameter():
    p = evm.Parameter(value=1.0, lower=0.0, upper=2.0)
    assert p.value == 1.0
    assert p.lower == 0.0
    assert p.upper == 2.0
    assert p.boundary_constraint == 0.0
    assert p.prior is None


def test_NormalParameter():
    p = evm.NormalParameter(value=1.0, lower=0.0, upper=2.0)
    assert p.value == 1.0
    assert p.lower == 0.0
    assert p.upper == 2.0
    assert p.boundary_constraint == 0.0
    assert isinstance(p.prior, Normal)
