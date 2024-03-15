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
    p = evm.Parameter(value=jnp.array(1.0))
    u = evm.effect.unconstrained()

    assert u.constraint(p) == Flat()
    assert u.scale_factor(p, jnp.array([1.0])) == SF(
        multiplicative=jnp.array([1.0]), additive=jnp.array([0.0])
    )


def test_gauss():
    p = evm.Parameter(value=jnp.array(0.0))
    g = evm.effect.gauss(width=jnp.array(1.0))

    assert isinstance(g.constraint(p), Gauss)
    assert g.scale_factor(p, jnp.array([1.0])) == SF(
        multiplicative=jnp.array([1.0]), additive=jnp.array([0.0])
    )


def test_lnN():
    p = evm.Parameter(value=jnp.array(0.0))
    ln = evm.effect.lnN(up=jnp.array([1.1]), down=jnp.array([0.9]))

    assert isinstance(ln.constraint(p), Gauss)
    assert ln.scale_factor(p, jnp.array([1.0])) == SF(
        multiplicative=jnp.array([1.0]), additive=jnp.array([0.0])
    )


def test_poisson():
    p = evm.Parameter(value=jnp.array(0.0))
    po = evm.effect.poisson(lamb=jnp.array(10))

    assert isinstance(po.constraint(p), Poisson)
    # assert po.scale_factor(p, jnp.array(1.0)) == pytest.approx(1.0) # FIXME
    # assert po.scale_factor(p.update(jnp.array(2.0)), jnp.array(1.0)) == pytest.approx(1.1) # FIXME


def test_shape():
    pass


def test_modifier():
    mu = evm.Parameter(value=jnp.array(1.1))
    norm = evm.Parameter(value=jnp.array(0.0))

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
    mu = evm.Parameter(value=jnp.array(1.1))
    norm = evm.Parameter(value=jnp.array(0.0))

    # unconstrained effect
    m_unconstrained = evm.Modifier(parameter=mu, effect=evm.effect.unconstrained())
    # gauss effect
    m_gauss = evm.Modifier(parameter=norm, effect=evm.effect.gauss(jnp.array(0.1)))

    # compose
    m = evm.modifier.compose(m_unconstrained, m_gauss)

    assert len(m) == 2
    assert m(jnp.array([10])) == pytest.approx(11)
