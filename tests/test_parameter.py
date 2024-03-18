from __future__ import annotations

import jax.numpy as jnp
import pytest

import evermore as evm
from evermore.custom_types import SF, _NoValue
from evermore.pdf import Flat, Gauss, Poisson


def test_parameter():
    p = evm.Parameter(value=jnp.array(1.0), lower=jnp.array(0.0), upper=jnp.array(2.0))
    assert p.value == 1.0
    assert p.lower == 0.0
    assert p.upper == 2.0
    assert p.boundary_penalty == 0.0
    assert p.constraint is _NoValue


def test_unconstrained():
    p = evm.FreeFloating()
    u = evm.effect.unconstrained()

    assert p.is_valid_effect(u)
    assert p.constraint == Flat()
    assert u.scale_factor(p, jnp.array([1.0])) == SF(
        multiplicative=jnp.array([1.0]), additive=jnp.array([0.0])
    )


def test_gauss():
    p = evm.GaussConstrained()
    g = evm.effect.gauss(width=jnp.array(1.0))

    assert p.is_valid_effect(g)
    assert isinstance(p.constraint, Gauss)
    assert g.scale_factor(p, jnp.array([1.0])) == SF(
        multiplicative=jnp.array([1.0]), additive=jnp.array([0.0])
    )


def test_lnN():
    p = evm.GaussConstrained()
    ln = evm.effect.lnN(up=jnp.array([1.1]), down=jnp.array([0.9]))

    assert p.is_valid_effect(ln)
    assert p.constraint, Gauss
    assert ln.scale_factor(p, jnp.array([1.0])) == SF(
        multiplicative=jnp.array([1.0]), additive=jnp.array([0.0])
    )


def test_poisson():
    p = evm.PoissonConstrained(hist=jnp.array([10]))
    po = evm.effect.poisson(lamb=jnp.array(10))

    assert p.is_valid_effect(po)
    assert isinstance(p.constraint, Poisson)
    # assert po.scale_factor(p, jnp.array(1.0)) == pytest.approx(1.0) # FIXME
    # assert po.scale_factor(p.update(jnp.array(2.0)), jnp.array(1.0)) == pytest.approx(1.1) # FIXME


def test_shape():
    pass


def test_modifier():
    mu = evm.FreeFloating(value=jnp.array(1.1))
    norm = evm.GaussConstrained(value=jnp.array(0.0))

    # unconstrained effect
    m_unconstrained = evm.Modifier(parameter=mu, effect=evm.effect.unconstrained())
    assert m_unconstrained(jnp.array([10])) == pytest.approx(11)

    # gauss effect
    m_gauss = evm.Modifier(parameter=norm, effect=evm.effect.gauss(jnp.array(0.1)))
    assert m_gauss(jnp.array([10])) == pytest.approx(10)

    # lnN effect
    m_lnN = evm.Modifier(
        parameter=norm,
        effect=evm.effect.lnN(up=jnp.array([1.1]), down=jnp.array([0.9])),
    )
    assert m_lnN(jnp.array([10])) == pytest.approx(10)

    # poisson effect # FIXME
    # m_poisson = Modifier(name="norm", parameter=norm, effect=poisson(jnp.array(10)))
    # assert m_poisson(jnp.array(10)) == pytest.approx(10)

    # shape effect # FIXME
    # effect = shape(up=jnp.array(12), down=jnp.array(8))
    # m_shape = Modifier(name="norm", parameter=norm, effect=effect)
    # assert m_shape(jnp.array(10)) == pytest.approx(10)


def test_compose():
    mu = evm.FreeFloating(value=jnp.array(1.1))
    norm = evm.GaussConstrained(value=jnp.array(0.0))

    # unconstrained effect
    m_unconstrained = evm.Modifier(parameter=mu, effect=evm.effect.unconstrained())
    # gauss effect
    m_gauss = evm.Modifier(parameter=norm, effect=evm.effect.gauss(jnp.array(0.1)))

    # compose
    m = evm.modifier.compose(m_unconstrained, m_gauss)

    assert len(m) == 2
    assert m(jnp.array([10])) == pytest.approx(11)
