from __future__ import annotations

from collections.abc import Hashable
from typing import Callable

import equinox as eqx
import jax
import jaxopt


class JaxOptimizer(eqx.Module):
    """
    Wrapper around `jaxopt` optimizers to make them hashable.
    This allows to pass the optimizer as a parameter to a `jax.jit` function, and setup the optimizer therein.

    Example:
    ```
        optimizer = JaxOptimizer.make(
            name="ScipyMinimize",
            settings={"method": "trust-constr"},
        )

        # or

        optimizer = JaxOptimizer.make(
            name="LBFGS",
            settings={
                "maxiter": 30,
                "tol": 1e-6,
                "jit": True,
                "unroll": True,
            },
        )

    ```
    """

    name: str
    _settings: tuple[tuple[str, Hashable], ...]

    def __init__(self, name: str, _settings: tuple[tuple[str, Hashable], ...]) -> None:
        self.name = name
        self._settings = _settings

    @classmethod
    def make(
        cls: type[JaxOptimizer], name: str, settings: dict[str, Hashable]
    ) -> JaxOptimizer:
        return cls(name=name, _settings=tuple(settings.items()))

    @property
    def settings(self) -> dict[str, Hashable]:
        return dict(self._settings)

    def solver_instance(self, fun: Callable) -> jaxopt._src.base.Solver:
        return getattr(jaxopt, self.name)(fun=fun, **self.settings)

    def fit(self, fun: Callable, init_values: dict[str, float]) -> jax.Array:
        values, state = self.solver_instance(fun=fun).run(init_values)
        return values, state
