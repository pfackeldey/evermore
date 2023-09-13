from __future__ import annotations

import abc

import jax
import jax.numpy as jnp

from dilax.parameter import Parameter

import equinox as eqx


class EvaluationResult(eqx.Module):
    """
    Result of the `expectation` method of a `Model`.

    Returns:
        dict[str, jax.Array]: The expected number of events in each bin for each process.
        jax.Array: The penalty term for the NLL.
    """

    expectations: dict[str, jax.Array]
    penalty: jax.Array

    def __init__(
        self,
        expectations: dict[str, jax.Array],
        penalty: jax.Array,
    ):
        self.expectations = expectations
        self.penalty = penalty

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(expectations={self.expectations}, penalty={self.penalty})"
        )

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

    processes: dict[str, jax.Array]
    parameters: dict[str, Parameter]

    def __init__(
        self,
        processes: dict[str, jax.Array],
        parameters: dict[str, Parameter],
    ):
        self.processes = processes
        self.parameters = parameters

    @property
    def n_processes(self) -> int:
        return len(self.processes)

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    def repr(self, detail_level: int = 0) -> str:
        if detail_level <= 0:
            args = f"({self.n_processes} processes, {self.n_parameters} parameters)"
        if detail_level > 0:
            args = f"(processes={list(self.processes.keys())}, parameters={list(self.parameters.keys())})"
        repr = f"{self.__class__.__name__}{args}"
        if detail_level > 1:
            if self.processes:
                repr += "\n\nProcesses:"
                for name, sumw in self.processes.items():
                    repr += f"\n  - {name}: sumw={sumw}"
            if self.parameters:
                repr += "\n\nParameters:"
                for name, param in self.parameters.items():
                    repr += f"\n  - {name}: {param}"
        return repr

    def __repr__(self) -> str:
        return self.repr(detail_level=1)

    @property
    def parameter_values(self) -> dict[str, float]:
        # avoid 0-dim arrays
        return {key: param.value for key, param in self.parameters.items()}

    def update(
        self,
        processes: dict[str, jax.Array] = {},
        values: dict[str, jax.Array] = {},
    ) -> Model:
        def _patch_processes(processes: dict[str, jax.Array]) -> dict[str, jax.Array]:
            # replace processes
            new_processes = dict(self.processes)
            for key, process in new_processes.items():
                if key in processes:
                    new_processes[key] = processes[key]
            return new_processes

        def _patch_parameters(values: dict[str, jax.Array]) -> dict[str, jax.Array]:
            # replace parameters
            new_parameters = {key: param for key, param in self.parameters.items()}
            for key, parameter in new_parameters.items():
                if key in values:
                    new_parameters[key] = parameter.update(value=values[key])
            return new_parameters

        return self.__class__(
            processes=_patch_processes(processes),
            parameters=_patch_parameters(values),
        )

    def nll_boundary_penalty(self) -> jax.Array:
        penalty = jnp.array(0.0)

        for param in self.parameters.values():
            penalty += param.boundary_penalty

        return penalty

    @abc.abstractmethod
    def evaluate(self) -> EvaluationResult:
        ...
