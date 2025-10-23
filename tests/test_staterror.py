from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import evermore as evm

jax.config.update("jax_enable_x64", True)


def test_staterrors_preserves_hist_for_default_values():
    hist = jnp.array([10.0, 5.0])
    variance = jnp.array([4.0, 0.0])

    staterrors = evm.staterror.StatErrors(hist, variance)

    np.testing.assert_allclose(staterrors(hist), hist)


def test_staterrors_masks_empty_bins():
    hist = jnp.array([10.0, 5.0])
    variance = jnp.array([4.0, 0.0])

    staterrors = evm.staterror.StatErrors(hist, variance)
    # apply a +0.5 sigma shift
    staterrors.parameter.value = jnp.array([0.5, 0.5])

    modified = staterrors(hist)

    # first bin receives the shift, second bin remains unchanged (masked)
    expected_first_bin = hist[0] * (1 + 0.5 * 0.2)  # relative error = 1/sqrt(25) = 0.2
    np.testing.assert_allclose(modified[0], expected_first_bin)
    np.testing.assert_allclose(modified[1], hist[1])
