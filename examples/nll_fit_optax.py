import equinox as eqx
import optax
import wadler_lindig as wl
from jaxtyping import PyTree
from model import Hist1D, hists, loss, observation, params

import evermore as evm

optim = optax.sgd(learning_rate=1e-2)


@eqx.filter_jit
def make_step(
    dynamic: PyTree[evm.Parameter],
    static: PyTree[evm.Parameter],
    opt_state: PyTree,
    hists: PyTree[Hist1D],
    observation: Hist1D,
) -> tuple[PyTree[evm.Parameter], PyTree]:
    grads = eqx.filter_grad(loss)(dynamic, static, hists, observation)
    updates, opt_state = optim.update(grads, opt_state)
    # apply parameter updates
    dynamic = eqx.apply_updates(dynamic, updates)
    return dynamic, opt_state


def fit(params, hists, observation):
    dynamic, static = evm.tree.partition(params)

    # initialize optimizer state
    opt_state = optim.init(eqx.filter(dynamic, eqx.is_inexact_array))

    # minimize params with 5000 steps
    for step in range(5000):
        if step % 500 == 0:
            loss_val = loss(dynamic, static, hists, observation)
            print(f"{step=} - {loss_val=:.6f}")
        dynamic, opt_state = make_step(dynamic, static, opt_state, hists, observation)

    # combine optimized dynamic part with the static pytree
    return evm.tree.combine(dynamic, static)


if __name__ == "__main__":
    bestfit_params = fit(params, hists, observation)

    print("Bestfit parameter:")
    wl.pprint(evm.tree.pure(bestfit_params), short_arrays=False)
