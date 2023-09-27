from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from dilax.parameter import Parameter
from dilax.util import FrozenDB, HistDB


class Result(eqx.Module):
    """
    Holds:
        dict[str, jax.Array]: The expected number of events in each bin for each process.
    """

    expectations: dict[str, jax.Array]

    def __init__(self, expectations: dict[str, jax.Array] = {}) -> None:
        self.expectations = expectations

    def add(self, process: str, expectation: jax.Array) -> Result:
        self.expectations[process] = expectation
        return self

    def expectation(self) -> jax.Array:
        expectation = jnp.array(0.0)
        for _, sumw in self.expectations.items():
            expectation += sumw
        return expectation


class Model(eqx.Module):
    """
    A model describing nuisance parameters, templates (histograms), and how they interact.
    It is requires to implement the `evaluate` method, which returns an `EvaluationResult` object.

    Example:
    ```
        # Simple model with two processes and two parameters

        class MyModel(Model):
            def evaluate(self) -> EvaluationResult:
                expectations = {}
                # signal
                signal, mu_penalty = self.parameters["mu"](self.processes["signal"], type="r")
                expectations["signal"] = signal

                # background
                background, sigma_penalty = self.parameters["sigma"](self.processes["background"], type="lnN", width=1.1)
                expectations["background"] = background
                return EvaluationResult(expectations=expectations, penalty=mu_penalty + sigma_penalty)


        model = MyModel(
            processes={"signal": jnp.array([10]), "background": jnp.array([50])},
            parameters={"mu": Parameter(value=1.0, bounds=(0, 100)), "sigma": Parameter(value=0, bounds=(-100, 100))},
        )

        # evaluate the expectation
        model.evaluate().expectation()
        >> Array([60.], dtype=float32, weak_type=True)

        %timeit model.evaluate().expectation()
        >> 245 µs ± 1.17 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        # evaluate the expectation *fast*
        def eval(model) -> jax.Array:
            res = model.evaluate()
            return res.expectation()

        eqx.filter_jit(eval)(model)
        >> Array([60.], dtype=float32, weak_type=True)

        %timeit eqx.filter_jit(eval)(model).block_until_ready()
        >> 96.9 µs ± 778 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    ```
    """

    processes: HistDB
    parameters: dict[str, Parameter]

    def __init__(self, processes: HistDB, parameters: dict[str, Parameter]) -> None:
        self.processes = processes
        self.parameters = parameters

    @property
    def parameter_values(self) -> dict[str, jax.Array]:
        return {key: param.value for key, param in self.parameters.items()}

    def parameter_constraints(self) -> jax.Array:
        c = []
        for param in self.parameters.values():
            # skip if the parameter was not used / has no constraint
            if not param.constraints:
                continue
            if not len(param.constraints) <= 1:
                msg = f"More than one constraint per parameter is not allowed. Got: {param.constraint}"
                raise ValueError(msg)
            constraint = next(iter(param.constraints))
            c.append(constraint.logpdf(param.value))
        return jnp.sum(jnp.array(c))

    def update(
        self, processes: dict | HistDB = {}, values: dict[str, jax.Array] = {}
    ) -> Model:
        if not isinstance(processes, HistDB):
            processes = HistDB(processes)

        def _patch_processes(processes: HistDB) -> HistDB:
            assert isinstance(processes, HistDB)
            new_processes = dict(self.processes.items())
            for key, _process in new_processes.items():
                if (key := FrozenDB.keyify(key)) in processes:
                    new_processes[key] = processes[key]
            return HistDB(new_processes)

        def _patch_parameters(values: dict[str, jax.Array]) -> dict[str, Parameter]:
            # replace parameters
            new_parameters = dict(self.parameters)
            for key, parameter in new_parameters.items():
                if key in values:
                    new_parameters[key] = parameter.update(value=values[key])
            return new_parameters

        return self.__class__(
            processes=_patch_processes(processes) if processes else self.processes,
            parameters=_patch_parameters(values) if values else self.parameters,
        )

    def nll_boundary_penalty(self) -> jax.Array:
        penalty = jnp.array([0.0])

        for param in self.parameters.values():
            penalty += param.boundary_penalty

        return penalty

    @abc.abstractmethod
    def __call__(self, processes: HistDB, parameters: dict[str, Parameter]) -> Result:
        ...

    def evaluate(self) -> Result:
        # evaluate the model with its current state
        return self(self.processes, self.parameters)
