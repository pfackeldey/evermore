import typing as tp

import equinox as eqx
import iminuit
import jax
import jax.numpy as jnp
import wadler_lindig as wl
from jaxtyping import Array, Float, PyTree
from model import hists, loss, observation, params

import evermore as evm

FlatV: tp.TypeAlias = Float[Array, " nparams"]  # type: ignore[name-defined]


def fit(params, hists, observation):
    # partition into dynamic and static parts
    dynamic, static = evm.tree.partition(params)

    # flatten parameter.value for iminuit
    values = evm.tree.pure(dynamic)
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)

    def update(
        params: PyTree[evm.Parameter],
        values: FlatV,
    ) -> PyTree[evm.Parameter]:
        return jax.tree.map(
            evm.parameter.replace_value,
            params,
            unravel_fn(values),
            is_leaf=evm.filter.is_parameter,
        )

    # wrap loss that works on flat array
    @eqx.filter_jit
    def flat_loss(flat_values: FlatV) -> Float[Array, ""]:
        _dynamic = update(dynamic, flat_values)
        return loss(_dynamic, static, hists, observation)

    minuit = iminuit.Minuit(flat_loss, flat_values, grad=eqx.filter_grad(flat_loss))
    minuit.errordef = iminuit.Minuit.LIKELIHOOD
    minuit.tol = 1e-5
    minuit.migrad()

    print(f"Is function minimum valid? -> {minuit.valid}")

    bestfit_values = jnp.array(minuit.values)

    # wrap into pytree of parameters
    bestfit_dynamic = update(dynamic, bestfit_values)

    # combine with static pytree
    return evm.tree.combine(bestfit_dynamic, static)


if __name__ == "__main__":
    bestfit_params = fit(params, hists, observation)

    print("Bestfit parameter:")
    wl.pprint(evm.tree.pure(bestfit_params), short_arrays=False)
