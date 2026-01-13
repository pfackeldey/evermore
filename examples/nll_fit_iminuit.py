import iminuit
import jax
import jax.numpy as jnp
import wadler_lindig as wl
from flax import nnx
from jaxtyping import Array, Float
from model import hists, loss, make_model, observation

import evermore as evm


def fit(model, hists, observation):
    # partition into dynamic and static parts
    graphdef, dynamic, static = nnx.split(model, evm.filter.is_dynamic_parameter, ...)
    args = (graphdef, static, hists, observation)

    # flatten parameter.get_value()(s) for iminuit
    values = nnx.pure(dynamic)
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)  # ty:ignore[possibly-missing-attribute]

    # wrap loss that works on flat array
    @nnx.jit
    def iminuit_loss(flat_values: Float[Array, " nparams"]) -> Float[Array, ""]:
        dynamic.replace_by_pure_dict(unravel_fn(flat_values))
        return loss(dynamic, args)

    minuit = iminuit.Minuit(iminuit_loss, flat_values, grad=nnx.grad(iminuit_loss))  # ty:ignore[invalid-argument-type]
    minuit.errordef = iminuit.Minuit.LIKELIHOOD
    minuit.tol = 1e-5
    minuit.migrad()

    print(f"Is function minimum valid? -> {minuit.valid}")

    # update dynamic part with bestfit values
    dynamic.replace_by_pure_dict(unravel_fn(jnp.array(minuit.values)))
    return nnx.merge(graphdef, dynamic, static)


if __name__ == "__main__":
    model = make_model()
    bestfit_params = fit(model, hists, observation)

    print("Bestfit parameter:")
    wl.pprint(nnx.pure(bestfit_params), short_arrays=False)
