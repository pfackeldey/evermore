from __future__ import annotations

import abc
from typing import TYPE_CHECKING, cast

import equinox as eqx
import jax
import jax.numpy as jnp

from dilax.parameter import Parameter
from dilax.util import FrozenDB, HistDB, Sentinel, _NoValue


class Result(eqx.Module):
    expectations: dict[str, jax.Array]

    def __init__(self) -> None:
        self.expectations = {}

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

    .. code-block:: python

        import jax
        import jax.numpy as jnp
        import equinox as eqx

        from dilax.model import Model, Result
        from dilax.parameter import Parameter, lnN, modifier, unconstrained
        from dilax.util import HistDB


        # Define a simple model with two processes and two parameters
        class MyModel(Model):
            def __call__(self, processes: HistDB, parameters: dict[str, Parameter]) -> Result:
                res = Result()

                # signal
                mu_mod = modifier(name="mu", parameter=parameters["mu"], effect=unconstrained())
                res.add(process="signal", expectation=mu_mod(processes["signal"]))

                # background
                bkg_mod = modifier(name="sigma", parameter=parameters["sigma"], effect=lnN(0.2))
                res.add(process="background", expectation=bkg_mod(processes["background"]))
                return res


        # Setup model
        processes = HistDB({"signal": jnp.array([10]), "background": jnp.array([50])})
        parameters = {
            "mu": Parameter(value=jnp.array([1.0]), bounds=(0.0, jnp.inf)),
            "sigma": Parameter(value=jnp.array([0.0])),
        }

        model = MyModel(processes=processes, parameters=parameters)

        # evaluate the expectation
        model.evaluate().expectation()
        # -> Array([60.], dtype=float32)

        %timeit model.evaluate().expectation()
        # -> 3.05 ms ± 29.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

        # evaluate the expectation *fast*
        @eqx.filter_jit
        def eval(model) -> jax.Array:
            res = model.evaluate()
            return res.expectation()

        eqx.filter_jit(eval)(model)
        # -> Array([60.], dtype=float32)

        %timeit eqx.filter_jit(eval)(model).block_until_ready()
        # -> 114 µs ± 327 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
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
        self,
        processes: dict | HistDB | Sentinel = _NoValue,
        values: dict[str, jax.Array] | Sentinel = _NoValue,
    ) -> Model:
        if values is _NoValue:
            values = {}
        if processes is _NoValue:
            processes = {}
        if not isinstance(processes, HistDB):
            processes = HistDB(processes)

        if TYPE_CHECKING:
            values = cast(dict[str, jax.Array], values)

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
