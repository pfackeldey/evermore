from __future__ import annotations

from functools import partial

import jax
import jaxopt
import pytest

from dilax.optimizer import JaxOptimizer


def test_jaxoptimizer():
    opt = JaxOptimizer.make(name="GradientDescent", settings={"maxiter": 5})

    assert opt.name == "GradientDescent"
    assert opt.settings == {"maxiter": 5}

    assert isinstance(opt.solver_instance(fun=lambda x: x), jaxopt.GradientDescent)

    # jit compatibility
    @partial(jax.jit, static_argnums=0)
    def f(optimizer):
        @jax.jit
        def fun(x):
            return (x - 2.0) ** 2

        init_values = 1.0
        values, _ = optimizer.fit(fun=fun, init_values=init_values)
        return values

    assert f(opt) == pytest.approx(2.0)
