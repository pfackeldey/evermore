from __future__ import annotations

import jax.numpy as jnp

from evermore import Parameter
from evermore.custom_types import OffsetAndScale
from evermore.effect import (
    AsymmetricExponential,
    Identity,
    Linear,
    VerticalTemplateMorphing,
)


def test_Identity():
    effect = Identity()

    assert effect(
        parameter=Parameter(value=0.0), hist=jnp.array([1, 2, 3])
    ) == OffsetAndScale(offset=0.0, scale=1.0)
    assert effect(
        parameter=Parameter(value=1.0), hist=jnp.array([1, 2, 3])
    ) == OffsetAndScale(offset=0.0, scale=1.0)
    assert effect(
        parameter=(Parameter(), Parameter()), hist=jnp.array([1, 2, 3])
    ) == OffsetAndScale(offset=0.0, scale=1.0)


def test_Linear():
    effect = Linear(slope=1.0, offset=0.0)
    assert effect(
        parameter=Parameter(value=0.0), hist=jnp.array([1, 2, 3])
    ) == OffsetAndScale(offset=0.0, scale=0.0)
    assert effect(
        parameter=Parameter(value=1.0), hist=jnp.array([1, 2, 3])
    ) == OffsetAndScale(offset=0.0, scale=1.0)

    effect = Linear(slope=0.0, offset=1.0)
    assert effect(
        parameter=Parameter(value=0.0), hist=jnp.array([1, 2, 3])
    ) == OffsetAndScale(offset=0.0, scale=1.0)
    assert effect(
        parameter=Parameter(value=1.0), hist=jnp.array([1, 2, 3])
    ) == OffsetAndScale(offset=0.0, scale=1.0)

    effect = Linear(slope=1.0, offset=1.0)
    assert effect(
        parameter=Parameter(value=0.0), hist=jnp.array([1, 2, 3])
    ) == OffsetAndScale(offset=0.0, scale=1.0)
    assert effect(
        parameter=Parameter(value=1.0), hist=jnp.array([1, 2, 3])
    ) == OffsetAndScale(offset=0.0, scale=2.0)


def test_AsymmetricExponential():
    effect = AsymmetricExponential(up=1.2, down=0.9)

    assert effect(
        parameter=Parameter(value=0.0), hist=jnp.array([1])
    ) == OffsetAndScale(offset=0.0, scale=1.0)
    assert effect(
        parameter=Parameter(value=+1.0), hist=jnp.array([1])
    ) == OffsetAndScale(offset=0.0, scale=1.2)
    assert effect(
        parameter=Parameter(value=-1.0), hist=jnp.array([1])
    ) == OffsetAndScale(offset=0.0, scale=0.9)


def test_VerticalTemplateMorphing():
    effect = VerticalTemplateMorphing(
        up_template=jnp.array([12]), down_template=jnp.array([7])
    )

    assert effect(
        parameter=Parameter(value=0.0), hist=jnp.array([10])
    ) == OffsetAndScale(offset=0.0, scale=1.0)
    assert effect(
        parameter=Parameter(value=+1.0), hist=jnp.array([10])
    ) == OffsetAndScale(offset=2.0, scale=1.0)
    assert effect(
        parameter=Parameter(value=-1.0), hist=jnp.array([10])
    ) == OffsetAndScale(offset=-3.0, scale=1.0)
