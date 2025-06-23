import equinox as eqx
import iminuit
import jax
import jax.numpy as jnp
import wadler_lindig as wl
from jaxtyping import Array, Float, PyTree
from model import hists, model, observation, params

import evermore as evm

jax.config.update("jax_enable_x64", True)


def update_params(
    params: PyTree[evm.Parameter], values: [PyTree[Array]]
) -> PyTree[evm.Parameter]:
    return jax.tree.map(
        evm.parameter.replace_value,
        params,
        values,
        is_leaf=evm.parameter.is_parameter,
    )


@eqx.filter_jit
def loss(
    dynamic: PyTree[evm.Parameter],
    static: PyTree[evm.Parameter],
    hists: PyTree[Float[Array, " nbins"]],
    observation: Float[Array, " nbins"],
) -> Float[Array, ""]:
    params = evm.parameter.combine(dynamic, static)
    expectations = model(params, hists)
    constraints = evm.loss.get_log_probs(params)
    loss_val = (
        evm.pdf.PoissonContinuous(evm.util.sum_over_leaves(expectations))
        .log_prob(observation)
        .sum()
    )
    # add constraint
    loss_val += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(loss_val)


def fit(params, hists, observation):
    # partition into dynamic and static parts
    dynamic, static = evm.parameter.partition(params)

    # flatten parameter.value for iminuit
    values = jax.tree.map(
        lambda p: p.value, dynamic, is_leaf=evm.parameter.is_parameter
    )
    flat_values, unravel_fn = jax.flatten_util.ravel_pytree(values)

    # wrap loss that works on flat array
    @eqx.filter_jit
    def flat_loss(flat_values: Float[Array, "..."]) -> Float[Array, ""]:
        param_values = unravel_fn(flat_values)
        _dynamic = update_params(dynamic, param_values)
        return loss(_dynamic, static, hists, observation)

    minuit = iminuit.Minuit(flat_loss, flat_values, grad=eqx.filter_grad(flat_loss))
    minuit.errordef = iminuit.Minuit.LIKELIHOOD
    minuit.tol = 1e-5
    minuit.migrad()

    print(f"Is function minimum valid? -> {minuit.valid}")

    bestfit = jnp.array(minuit.values)

    # wrap into pytree
    bestfit_wrapped = unravel_fn(bestfit)

    # wrap into pytree of parameters
    bestfit_dynamic = update_params(dynamic, bestfit_wrapped)

    # combine with static pytree
    return evm.parameter.combine(bestfit_dynamic, static)


if __name__ == "__main__":
    bestfit_params = fit(params, hists, observation)

    print("Bestfit parameter:")
    wl.pprint(bestfit_params, short_arrays=False)
