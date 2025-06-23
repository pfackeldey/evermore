import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import wadler_lindig as wl
from jaxtyping import Array, Float, PyTree
from model import hists, model, observation, params

import evermore as evm

jax.config.update("jax_enable_x64", True)
optim = optax.sgd(learning_rate=1e-2)


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


@eqx.filter_jit
def make_step(
    dynamic: PyTree[evm.Parameter],
    static: PyTree[evm.Parameter],
    opt_state: PyTree,
    hists: PyTree[Float[Array, " nbins"]],
    observation: Float[Array, " nbins"],
) -> tuple[PyTree[evm.Parameter], PyTree]:
    grads = eqx.filter_grad(loss)(dynamic, static, hists, observation)
    updates, opt_state = optim.update(grads, opt_state)
    # apply parameter updates
    dynamic = eqx.apply_updates(dynamic, updates)
    return dynamic, opt_state


def fit(params, hists, observation):
    dynamic, static = evm.parameter.partition(params)

    # initialize optimizer state
    opt_state = optim.init(eqx.filter(dynamic, eqx.is_inexact_array))

    # minimize params with 5000 steps
    for step in range(5000):
        if step % 500 == 0:
            loss_val = loss(dynamic, static, hists, observation)
            print(f"{step=} - {loss_val=:.6f}")
        dynamic, opt_state = make_step(dynamic, static, opt_state, hists, observation)

    # combine optimized dynamic part with the static pytree
    return evm.parameter.combine(dynamic, static)


if __name__ == "__main__":
    bestfit_params = fit(params, hists, observation)

    print("Bestfit parameter:")
    wl.pprint(bestfit_params, short_arrays=False)
