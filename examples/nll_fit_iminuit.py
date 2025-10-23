import typing as tp

import iminuit
import jax
import jax.numpy as jnp
import wadler_lindig as wl
from flax import nnx
from jaxtyping import Array, Float
from model import hists, loss, observation, params

import evermore as evm

FlatV: tp.TypeAlias = Float[Array, " nparams"]  # type: ignore[name-defined]


def fit(params, hists, observation):
    # partition into dynamic and static parts
    graphdef, dynamic, static = nnx.split(params, evm.filter.is_dynamic_parameter, ...)

    # flatten parameter.value(s) for iminuit
    values = nnx.pure(dynamic)
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)

    # wrap loss that works on flat array
    @nnx.jit
    def iminuit_loss(flat_values: FlatV) -> Float[Array, ""]:
        dynamic.replace_by_pure_dict(unravel_fn(flat_values))
        params = nnx.merge(graphdef, dynamic, static, copy=True)
        return loss(params, hists=hists, observation=observation)

    minuit = iminuit.Minuit(iminuit_loss, flat_values, grad=nnx.grad(iminuit_loss))
    minuit.errordef = iminuit.Minuit.LIKELIHOOD
    minuit.tol = 1e-5
    minuit.migrad()

    print(f"Is function minimum valid? -> {minuit.valid}")

    # update dynamic part with bestfit values
    dynamic.replace_by_pure_dict(unravel_fn(jnp.array(minuit.values)))
    return nnx.merge(graphdef, dynamic, static)


if __name__ == "__main__":
    bestfit_params = fit(params, hists, observation)

    print("Bestfit parameter:")
    wl.pprint(nnx.pure(bestfit_params), short_arrays=False)
