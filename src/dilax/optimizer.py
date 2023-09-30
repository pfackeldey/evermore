from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Callable, cast

import equinox as eqx
import jax
import jaxopt

from dilax.util import Sentinel, _NoValue


class JaxOptimizer(eqx.Module):
    """
    Wrapper around `jaxopt` optimizers to make them hashable.
    This allows to pass the optimizer as a parameter to a `jax.jit` function, and setup the optimizer therein.

    Example:

    .. code-block:: python

        optimizer = JaxOptimizer.make(name="GradientDescent", settings={"maxiter": 5})
        # or, e.g.: optimizer = JaxOptimizer.make(name="LBFGS", settings={"maxiter": 10})

        optimizer.fit(fun=nll, init_values=init_values)
    """

    name: str
    _settings: tuple[tuple[str, Hashable], ...]

    def __init__(self, name: str, _settings: tuple[tuple[str, Hashable], ...]) -> None:
        self.name = name
        self._settings = _settings

    @classmethod
    def make(
        cls: type[JaxOptimizer],
        name: str,
        settings: dict[str, Hashable] | Sentinel = _NoValue,
    ) -> JaxOptimizer:
        if settings is _NoValue:
            settings = {}
        if TYPE_CHECKING:
            settings = cast(dict[str, Hashable], settings)
        return cls(name=name, _settings=tuple(settings.items()))

    @property
    def settings(self) -> dict[str, Hashable]:
        return dict(self._settings)

    def solver_instance(self, fun: Callable) -> jaxopt._src.base.Solver:
        return getattr(jaxopt, self.name)(fun=fun, **self.settings)

    def fit(
        self, fun: Callable, init_values: dict[str, jax.Array]
    ) -> tuple[dict[str, jax.Array], Any]:
        values, state = self.solver_instance(fun=fun).run(init_values)
        return values, state


class Chain(eqx.Module):
    """
    Chain multiple optimizers together.
    They probably should have the `maxiter` setting set to a value,
    in order to have a deterministic runtime behaviour.

    Example:

    .. code-block:: python

        opt1 = JaxOptimizer.make(name="GradientDescent", settings={"maxiter": 5})
        opt2 = JaxOptimizer.make(name="LBFGS", settings={"maxiter": 10})

        chain = Chain(opt1, opt2)
        # first 5 steps are minimized with GradientDescent, then 10 steps with LBFGS
        chain.fit(fun=nll, init_values=init_values)
    """

    optimizers: tuple[JaxOptimizer, ...]

    def __init__(self, *optimizers: JaxOptimizer) -> None:
        self.optimizers = optimizers

    def fit(
        self, fun: Callable, init_values: dict[str, jax.Array]
    ) -> tuple[dict[str, jax.Array], Any]:
        values = init_values
        for optimizer in self.optimizers:
            values, state = optimizer.fit(fun=fun, init_values=values)
        return values, state
