from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from dilax.custom_types import Sentinel, _NoValue
from dilax.parameter import Parameter
from dilax.util import deep_update

__all__ = [
    "Result",
    "Model",
]


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
        return cast(jax.Array, sum(jtu.tree_leaves(self.expectations)))


def _is_parameter(leaf: Any) -> bool:
    return isinstance(leaf, Parameter)


def _is_none_or_is_parameter(leaf: Any) -> bool:
    return leaf is None or _is_parameter(leaf)


class Model(eqx.Module):
    """
    A model describing nuisance parameters, templates (histograms), and how they interact.
    It is requires to implement the `evaluate` method, which returns an `Result` object.

    Example:

    .. code-block:: python

        import equinox as eqx
        import jax
        import jax.numpy as jnp

        import dilax as dlx


        # Define a simple model with two processes and two parameters
        class MyModel(dlx.Model):
            def __call__(self, processes: dict, parameters: dict) -> dlx.Result:
                res = dlx.Result()

                # signal
                mu_mod = dlx.modifier(name="mu", parameter=parameters["mu"], effect=dlx.effect.unconstrained())
                res.add(process="signal", expectation=mu_mod(processes["signal"]))

                # background
                bkg_mod = dlx.modifier(name="sigma", parameter=parameters["sigma"], effect=dlx.effect.lnN(0.2))
                res.add(process="background", expectation=bkg_mod(processes["background"]))
                return res


        # Setup model
        processes = {"signal": jnp.array([10]), "background": jnp.array([50])}
        parameters = {
            "mu": dlx.Parameter(value=jnp.array([1.0]), bounds=(0.0, jnp.inf)),
            "sigma": dlx.Parameter(value=jnp.array([0.0])),
        }

        model = MyModel(processes=processes, parameters=parameters)

        # evaluate the expectation
        model.evaluate().expectation()
        # -> Array([60.], dtype=float32)

        %timeit model.evaluate().expectation()
        # -> 485 µs ± 1.49 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        # evaluate the expectation *fast*
        @eqx.filter_jit
        def eval(model) -> jax.Array:
            res = model.evaluate()
            return res.expectation()

        eqx.filter_jit(eval)(model)
        # -> Array([60.], dtype=float32)

        %timeit eqx.filter_jit(eval)(model).block_until_ready()
        # -> 202 µs ± 4.87 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    """

    processes: dict
    parameters: dict[str, Parameter]
    auxiliary: Any

    def __init__(
        self,
        processes: dict,
        parameters: dict,
        auxiliary: Any | Sentinel = _NoValue,
    ) -> None:
        self.processes = processes
        self.parameters = parameters
        if auxiliary is _NoValue:
            auxiliary = {}
        self.auxiliary = auxiliary

    @property
    def parameter_values(self) -> dict:
        return jtu.tree_map(
            lambda l: l.value,  # noqa: E741
            self.parameters,
            is_leaf=_is_parameter,
        )

    def parameter_constraints(self) -> dict:
        def _constraint(param: Parameter) -> jax.Array:
            if param.constraints:
                if len(param.constraints) > 1:
                    msg = f"More than one constraint per parameter is not allowed. Got: {param.constraint}"
                    raise ValueError(msg)
                return next(iter(param.constraints)).logpdf(param.value)
            return jnp.array([0.0])

        return jtu.tree_map(
            _constraint,
            self.parameters,
            is_leaf=_is_parameter,
        )

    def update(
        self,
        processes: dict | Sentinel = _NoValue,
        values: dict | Sentinel = _NoValue,
    ) -> Model:
        if values is _NoValue:
            values = {}
        if processes is _NoValue:
            processes = {}

        if TYPE_CHECKING:
            values = cast(dict, values)
            processes = cast(dict, processes)

        # patch original processes with new ones
        new_processes = deep_update(self.processes, processes)

        # patch original parameters with new ones
        _updates = deep_update(
            jtu.tree_map(lambda _: None, self.parameters, is_leaf=_is_parameter),
            values,
        )

        def _update_params(update: jax.Array | None, param: Parameter) -> Parameter:
            if update is None:
                return param
            return param.update(value=update)

        new_parameters = jtu.tree_map(
            _update_params,
            _updates,
            self.parameters,
            is_leaf=_is_none_or_is_parameter,
        )

        return eqx.tree_at(
            lambda t: (t.processes, t.parameters), self, (new_processes, new_parameters)
        )

    def nll_boundary_penalty(self) -> jax.Array:
        return cast(
            jax.Array,
            sum(
                jtu.tree_leaves(
                    jtu.tree_map(
                        lambda p: p.boundary_penalty,
                        self.parameters,
                        is_leaf=_is_parameter,
                    )
                )
            ),
        )

    @abc.abstractmethod
    def __call__(self, processes: dict, parameters: dict) -> Result:
        ...

    def evaluate(self) -> Result:
        # evaluate the model with its current state
        return self(self.processes, self.parameters)
