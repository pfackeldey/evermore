from __future__ import annotations

import jax.numpy as jnp
import pytest

from evermore.pdf import Flat, Normal, Poisson


def test_flat():
    pdf = Flat()

    assert pdf.log_prob(jnp.array(1.0)) == jnp.array(0.0)
    assert pdf.log_prob(jnp.array(2.0)) == jnp.array(0.0)
    assert pdf.log_prob(jnp.array(3.0)) == jnp.array(0.0)


def test_Normal():
    pdf = Normal(mean=jnp.array(0.0), width=jnp.array(1.0))

    assert pdf.log_prob(jnp.array(0.0)) == pytest.approx(0.0)


def test_poisson():
    pdf = Poisson(lamb=jnp.array(10))

    assert pdf.log_prob(jnp.array(-0.5)) == pytest.approx(-1.196003)
