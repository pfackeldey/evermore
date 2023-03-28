from __future__ import annotations

from typing import Callable, Hashable

import chex
import jax
import jaxopt

from dilax.model import Model


@chex.dataclass(frozen=True)
class JaxOptimizer:
    name: str
    _settings: tuple[tuple[str, Hashable], ...]

    @classmethod
    def make(cls: type[JaxOptimizer], name: str, settings: dict[str, Hashable]) -> JaxOptimizer:
        return cls(name=name, _settings=tuple(settings.items()))

    @property
    def settings(self) -> dict[str, Hashable]:
        return dict(self._settings)

    def solver_instance(self, fun: Callable) -> jaxopt._src.base.Solver:
        return getattr(jaxopt, self.name)(fun=fun, **self.settings)

    def fit(
        self,
        fun: Callable,
        init_params: dict[str, float],
        model: Model,
        observation: jax.Array,
    ) -> jax.Array:
        params, state = self.solver_instance(fun=fun).run(
            init_params, model=model, observation=observation
        )
        return params, state
