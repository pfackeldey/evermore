from __future__ import annotations

import jax.numpy as jnp
import pytest

import dilax as dlx
from dilax.pdf import Flat, Gauss, Poisson


def test_parameter():
    p = dlx.Parameter(value=jnp.array(1.0), bounds=(jnp.array(0.0), jnp.array(2.0)))

    assert p.value == 1.0
    assert p.update(jnp.array(2.0)).value == 2.0
    assert p.bounds == (0.0, 2.0)

    assert p.boundary_penalty == 0.0
    assert p.update(jnp.array(3.0)).boundary_penalty == jnp.inf


def test_unconstrained():
    p = dlx.Parameter(value=jnp.array(1.0))
    u = dlx.effect.unconstrained()

    assert u.constraint == Flat()
    assert u.scale_factor(p, jnp.array(1.0)) == pytest.approx(1.0)
    assert u.scale_factor(p.update(jnp.array(2.0)), jnp.array(1.0)) == pytest.approx(
        2.0
    )


def test_gauss():
    p = dlx.Parameter(value=jnp.array(0.0))
    g = dlx.effect.gauss(width=jnp.array(1.0))

    assert g.constraint == Gauss(mean=0.0, width=1.0)
    assert g.scale_factor(p, jnp.array(1.0)) == pytest.approx(1.0)
    assert g.scale_factor(p.update(jnp.array(2.0)), jnp.array(1.0)) == pytest.approx(
        3.0
    )


def test_lnN():
    p = dlx.Parameter(value=jnp.array(0.0))
    ln = dlx.effect.lnN(width=(0.9, 1.1))

    assert ln.constraint == Gauss(mean=0.0, width=1.0)
    assert ln.scale_factor(p, jnp.array(1.0)) == pytest.approx(1.0)
    # assert ln.scale_factor(p.update(jnp.array(2.0)), jnp.array(1.0)) == pytest.approx(1.1)


def test_poisson():
    # p = dlx.Parameter(value=jnp.array(0.0))
    po = dlx.effect.poisson(lamb=jnp.array(10))

    assert po.constraint == Poisson(lamb=jnp.array(10))
    # assert po.scale_factor(p, jnp.array(1.0)) == pytest.approx(1.0) # FIXME
    # assert po.scale_factor(p.update(jnp.array(2.0)), jnp.array(1.0)) == pytest.approx(1.1) # FIXME


def test_shape():
    pass


def test_modifier():
    mu = dlx.Parameter(value=jnp.array(1.1))
    norm = dlx.Parameter(value=jnp.array(0.0))

    # unconstrained effect
    m_unconstrained = dlx.modifier(
        name="mu", parameter=mu, effect=dlx.effect.unconstrained()
    )
    assert m_unconstrained(jnp.array([10])) == pytest.approx(11)

    # gauss effect
    m_gauss = dlx.modifier(
        name="norm", parameter=norm, effect=dlx.effect.gauss(jnp.array(0.1))
    )
    assert m_gauss(jnp.array([10])) == pytest.approx(10)

    # lnN effect
    m_lnN = dlx.modifier(
        name="norm", parameter=norm, effect=dlx.effect.lnN(width=(0.9, 1.1))
    )
    assert m_lnN(jnp.array([10])) == pytest.approx(10)

    # poisson effect # FIXME
    # m_poisson = modifier(name="norm", parameter=norm, effect=poisson(jnp.array(10)))
    # assert m_poisson(jnp.array(10)) == pytest.approx(10)

    # shape effect # FIXME
    # effect = shape(up=jnp.array(12), down=jnp.array(8))
    # m_shape = modifier(name="norm", parameter=norm, effect=effect)
    # assert m_shape(jnp.array(10)) == pytest.approx(10)


def test_compose():
    mu = dlx.Parameter(value=jnp.array(1.1))
    norm = dlx.Parameter(value=jnp.array(0.0))

    # unconstrained effect
    m_unconstrained = dlx.modifier(
        name="mu", parameter=mu, effect=dlx.effect.unconstrained()
    )
    # gauss effect
    m_gauss = dlx.modifier(
        name="norm", parameter=norm, effect=dlx.effect.gauss(jnp.array(0.1))
    )

    # compose
    m = dlx.compose(m_unconstrained, m_gauss)

    assert len(m) == 2
    assert m(jnp.array([10])) == pytest.approx(11)
