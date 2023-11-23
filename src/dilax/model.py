from __future__ import annotations

import abc
from typing import TYPE_CHECKING, cast

import equinox as eqx
import jax
import jax.numpy as jnp

from dilax.parameter import Parameter
from dilax.util import Sentinel, _NoValue

__all__ = ["Result", "Model"]


def __dir__():
    return __all__


class Result(eqx.Module):
    expectations: dict[str, jax.Array]

    def __init__(self) -> None:
        self.expectations = {}

    def add(self, process: str, expectation: jax.Array) -> Result:
        self.expectations[process] = expectation
        return self

    def expectation(self) -> jax.Array:
        return cast(jax.Array, sum(jax.tree_util.tree_leaves(self.expectations)))


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


        # Define a simple model with two processes and two parameters
        class MyModel(Model):
            def __call__(self, processes: dict, parameters: dict[str, Parameter]) -> Result:
                res = Result()

                # signal
                mu_mod = modifier(name="mu", parameter=parameters["mu"], effect=unconstrained())
                res.add(process="signal", expectation=mu_mod(processes["signal"]))

                # background
                bkg_mod = modifier(name="sigma", parameter=parameters["sigma"], effect=lnN(0.2))
                res.add(process="background", expectation=bkg_mod(processes["background"]))
                return res


        # Setup model
        processes = {"signal": jnp.array([10]), "background": jnp.array([50])}
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

    processes: dict
    parameters: dict[str, Parameter]

    def __init__(self, processes: dict, parameters: dict[str, Parameter]) -> None:
        self.processes = processes
        self.parameters = parameters

    @property
    def parameter_values(self) -> dict[str, jax.Array]:
        return {key: param.value for key, param in self.parameters.items()}

    def parameter_constraints(self) -> dict[str, jax.Array]:
        constraints = {}
        for name, param in self.parameters.items():
            # skip if the parameter was not used / has no constraint
            if not param.constraints:
                continue
            if not len(param.constraints) <= 1:
                msg = f"More than one constraint per parameter is not allowed. Got: {param.constraint}"
                raise ValueError(msg)
            constraint = next(iter(param.constraints))
            constraints[name] = constraint.logpdf(param.value)
        return constraints

    def update(
        self,
        processes: dict | Sentinel = _NoValue,
        values: dict[str, jax.Array] | Sentinel = _NoValue,
    ) -> Model:
        if values is _NoValue:
            values = {}
        if processes is _NoValue:
            processes = {}

        if TYPE_CHECKING:
            values = cast(dict[str, jax.Array], values)
            processes = cast(dict, processes)

        # patch original processes with new ones
        new_processes = {}
        for key, old_process in self.processes.items():
            if key in processes:
                new_process = processes[key]
                new_processes[key] = new_process
            else:
                new_processes[key] = old_process

        # patch original parameters with new ones
        new_parameters = {}
        for key, old_parameter in self.parameters.items():
            if key in values:
                new_parameter = old_parameter.update(value=values[key])
                new_parameters[key] = new_parameter
            else:
                new_parameters[key] = old_parameter

        return eqx.tree_at(
            lambda t: (t.processes, t.parameters), self, (new_processes, new_parameters)
        )

    def nll_boundary_penalty(self) -> jax.Array:
        penalty = jnp.array([0.0])

        for param in self.parameters.values():
            penalty += param.boundary_penalty

        return penalty

    @abc.abstractmethod
    def __call__(self, processes: dict, parameters: dict[str, Parameter]) -> Result:
        ...

    def evaluate(self) -> Result:
        # evaluate the model with its current state
        return self(self.processes, self.parameters)
