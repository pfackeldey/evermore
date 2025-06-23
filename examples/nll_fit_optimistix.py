import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import wadler_lindig as wl
from jaxtyping import Array, Float, PyTree
from model import hists, model, observation, params

import evermore as evm

jax.config.update("jax_enable_x64", True)


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
    solver = optx.BFGS(rtol=1e-5, atol=1e-7)

    dynamic, static = evm.parameter.partition(params)

    def optx_loss(dynamic, args):
        return loss(dynamic, *args)

    fitresult = optx.minimise(
        optx_loss,
        solver,
        dynamic,
        has_aux=False,
        args=(static, hists, observation),
        options={},
        max_steps=10_000,
        throw=True,
    )
    return evm.parameter.combine(fitresult.value, static)


if __name__ == "__main__":
    bestfit_params = fit(params, hists, observation)

    print("Bestfit parameter:")
    wl.pprint(bestfit_params, short_arrays=False)
