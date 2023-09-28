from __future__ import annotations

import jax.numpy as jnp
import pytest

from dilax.pdf import Flat, Gauss, Poisson


def test_flat():
    pdf = Flat()

    assert pdf.pdf(jnp.array(1.0)) == jnp.array(1.0)
    assert pdf.pdf(jnp.array(2.0)) == jnp.array(1.0)
    assert pdf.pdf(jnp.array(3.0)) == jnp.array(1.0)

    assert pdf.logpdf(jnp.array(1.0)) == jnp.array(0.0)
    assert pdf.logpdf(jnp.array(2.0)) == jnp.array(0.0)
    assert pdf.logpdf(jnp.array(3.0)) == jnp.array(0.0)


def test_gauss():
    pdf = Gauss(mean=0.0, width=1.0)

    assert pdf.pdf(jnp.array(0.0)) == pytest.approx(1.0 / jnp.sqrt(2 * jnp.pi))
    assert pdf.logpdf(jnp.array(0.0)) == pytest.approx(0.0)
    assert pdf.cdf(jnp.array(0.0)) == pytest.approx(0.5)
    assert pdf.inv_cdf(jnp.array(0.5)) == pytest.approx(0.0)
    assert pdf.inv_cdf(pdf.cdf(jnp.array(0.0))) == pytest.approx(0.0)


def test_poisson():
    pdf = Poisson(lamb=10)

    assert pdf.pdf(jnp.array(10)) == pytest.approx(0.12510978)
    assert pdf.logpdf(jnp.array(5)) == pytest.approx(-1.196003)
    assert pdf.cdf(jnp.array(10)) == pytest.approx(0.5830412)
    assert pdf.inv_cdf(jnp.array(0.5830412)) == pytest.approx(10)
    assert pdf.inv_cdf(pdf.cdf(jnp.array(10))) == pytest.approx(10)


def test_hashable():
    assert hash(Flat()) == hash(Flat())
    assert hash(Gauss(mean=0.0, width=1.0)) == hash(Gauss(mean=0.0, width=1.0))
    assert hash(Poisson(lamb=10)) == hash(Poisson(lamb=10))
