from __future__ import annotations

import jax.numpy as jnp
import pytest
from jaxtyping import Float, Scalar

from evermore.pdf import Normal, PoissonContinuous, PoissonDiscrete


def test_Normal():
    pdf: Normal[Float[Scalar, ""]] = Normal(mean=jnp.array(0.0), width=jnp.array(1.0))

    assert pdf.log_prob(jnp.array(0.0)) == pytest.approx(0.0)


def test_PoissonDiscrete():
    pdf: PoissonDiscrete[Float[Scalar, ""]] = PoissonDiscrete(lamb=jnp.array(10))

    assert pdf.log_prob(jnp.array(5.0)) == pytest.approx(-1.5342636)


def test_PoissonContinuous():
    pdf: PoissonContinuous[Float[Scalar, ""]] = PoissonContinuous(lamb=jnp.array(10))

    assert pdf.log_prob(jnp.array(5.0)) == pytest.approx(-1.5342636)
