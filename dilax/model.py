from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from dilax.parameter import Parameter


@chex.dataclass(frozen=True)
class Model:
    processes: dict[str, jax.Array]
    parameters: dict[str, Parameter]

    @property
    def parameter_strengths(self) -> dict[str, float]:
        return {key: param.strength for key, param in self.parameters.items()}

    def apply(
        self,
        processes: dict[str, jax.Array] = {},
        parameters: dict[str, jax.Array] = {},
    ) -> Model:
        # replace processes
        new_processes = dict(self.processes)
        for key, process in new_processes.items():
            if key in processes:
                new_processes[key] = processes[key]

        # replace parameters
        new_parameters = {key: param for key, param in self.parameters.items()}
        for key, parameter in new_parameters.items():
            if key in parameters:
                parameter.strength = parameters[key]
                new_parameters[key] = parameter
        return self.__class__(processes=new_processes, parameters=new_parameters)

    def nll_constraint(self) -> jax.Array:
        constraint = jnp.array(0.0)

        for param in self.parameters.values():
            constraint += param.logpdf
            constraint += param.boundary_constraint

        return constraint

    def eval(self) -> jax.Array:
        ...
