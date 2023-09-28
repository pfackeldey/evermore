from __future__ import annotations

import jax.numpy as jnp
import pytest

from dilax.parameter import (
    Parameter,
    compose,
    gauss,
    lnN,
    modifier,
    poisson,
    unconstrained,
)
from dilax.pdf import Flat, Gauss


def test_parameter():
    p = Parameter(value=jnp.array(1.0), bounds=(jnp.array(0.0), jnp.array(2.0)))

    assert p.value == 1.0
    assert p.update(jnp.array(2.0)).value == 2.0
    assert p.bounds == (0.0, 2.0)

    assert p.boundary_penalty == 0.0
    assert p.update(jnp.array(3.0)).boundary_penalty == jnp.inf


def test_unconstrained():
    p = Parameter(value=jnp.array(1.0))
    u = unconstrained()

    assert u.constraint == Flat()
    assert u.scale_factor(p, jnp.array(1.0)) == pytest.approx(1.0)
    assert u.scale_factor(p.update(jnp.array(2.0)), jnp.array(1.0)) == pytest.approx(
        2.0
    )


def test_gauss():
    p = Parameter(value=jnp.array(0.0))
    g = gauss(width=jnp.array(1.0))

    assert g.constraint == Gauss(mean=0.0, width=1.0)
    assert g.scale_factor(p, jnp.array(1.0)) == pytest.approx(1.0)
    assert g.scale_factor(p.update(jnp.array(2.0)), jnp.array(1.0)) == pytest.approx(
        3.0
    )


def test_lnN():
    p = Parameter(value=jnp.array(0.0))
    ln = lnN(width=jnp.array(0.1))

    assert ln.constraint == Gauss(mean=0.0, width=1.0)
    assert ln.scale_factor(p, jnp.array(1.0)) == pytest.approx(1.0)
    # assert ln.scale_factor(p.update(jnp.array(2.0)), jnp.array(1.0)) == pytest.approx(1.1)


def test_poisson():
    # p = Parameter(value=jnp.array(0.0))
    po = poisson(lamb=jnp.array(10))

    assert po.constraint == Gauss(mean=0.0, width=1.0)
    # assert po.scale_factor(p, jnp.array(1.0)) == pytest.approx(1.0) # FIXME
    # assert po.scale_factor(p.update(jnp.array(2.0)), jnp.array(1.0)) == pytest.approx(1.1) # FIXME


def test_shape():
    pass


def test_modifier():
    mu = Parameter(value=jnp.array(1.1))
    norm = Parameter(value=jnp.array(0.0))

    # unconstrained effect
    m_unconstrained = modifier(name="mu", parameter=mu, effect=unconstrained())
    assert m_unconstrained(jnp.array(10)) == pytest.approx(11)

    # gauss effect
    m_gauss = modifier(name="norm", parameter=norm, effect=gauss(jnp.array(0.1)))
    assert m_gauss(jnp.array(10)) == pytest.approx(10)

    # lnN effect
    m_lnN = modifier(name="norm", parameter=norm, effect=lnN(jnp.array(0.1)))
    assert m_lnN(jnp.array(10)) == pytest.approx(10)

    # poisson effect # FIXME
    # m_poisson = modifier(name="norm", parameter=norm, effect=poisson(jnp.array(10)))
    # assert m_poisson(jnp.array(10)) == pytest.approx(10)

    # shape effect # FIXME
    # effect = shape(up=jnp.array(12), down=jnp.array(8))
    # m_shape = modifier(name="norm", parameter=norm, effect=effect)
    # assert m_shape(jnp.array(10)) == pytest.approx(10)


def test_compose():
    mu = Parameter(value=jnp.array(1.1))
    norm = Parameter(value=jnp.array(0.0))

    # unconstrained effect
    m_unconstrained = modifier(name="mu", parameter=mu, effect=unconstrained())
    # gauss effect
    m_gauss = modifier(name="norm", parameter=norm, effect=gauss(jnp.array(0.1)))

    # compose
    m = compose(m_unconstrained, m_gauss)

    assert m.names == ["mu", "norm"]
    assert len(m) == 2
    assert m(jnp.array([10])) == pytest.approx(11)
