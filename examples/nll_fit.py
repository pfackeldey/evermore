import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree
from model import hists, model, observation, params

import evermore as evm

optim = optax.sgd(learning_rate=1e-2)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


@eqx.filter_jit
def loss(
    diffable: PyTree[evm.Parameter],
    static: PyTree[evm.Parameter],
    hists: PyTree[Float[Array, " nbins"]],
    observation: Float[Array, " nbins"],
) -> Float[Array, ""]:
    params = evm.parameter.combine(diffable, static)
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
    params: PyTree[evm.Parameter],
    opt_state: PyTree,
    hists: PyTree[Float[Array, " nbins"]],
    observation: Float[Array, " nbins"],
) -> tuple[PyTree[evm.Parameter], PyTree]:
    diffable, static = evm.parameter.partition(params)
    grads = eqx.filter_grad(loss)(diffable, static, hists, observation)
    updates, opt_state = optim.update(grads, opt_state)
    # apply parameter updates
    diffable = eqx.apply_updates(diffable, updates)
    params = evm.parameter.combine(diffable, static)
    return params, opt_state


# minimize params with 1000 steps
for step in range(1000):
    if step % 100 == 0:
        diffable, static = evm.parameter.partition(params)
        loss_val = loss(diffable, static, hists, observation)
        print(f"{step=} - {loss_val=:.6f}")
    params, opt_state = make_step(params, opt_state, hists, observation)


# For low overhead it is recommended to use jax.lax.fori_loop.
# In case you want to jit the for loop, you can use the following function,
# this will prevent jax from unrolling the loop and creating a huge graph
@jax.jit
def fit(steps: int = 1000) -> tuple[PyTree[evm.Parameter], PyTree]:
    def fun(step, params_optstate):
        params, opt_state = params_optstate
        return make_step(params, opt_state, hists, observation)

    return jax.lax.fori_loop(0, steps, fun, (params, opt_state))
