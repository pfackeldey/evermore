import equinox as eqx
import jax
import wadler_lindig as wl
from model import hists, loss, observation, params

import evermore as evm

evmm = evm.mutable
evmp = evm.parameter


def fit(params, hists, observation):
    dynamic, static = evmp.partition(params)

    # make dynamic part mutable
    dynamic_ref = evmm.mutable(dynamic)

    @jax.jit
    def minimize_step(dynamic_ref, static, hists, observation) -> None:
        loss_grad = eqx.filter_value_and_grad(loss)
        loss_val, grads = loss_grad(
            evmm.freeze(dynamic_ref),
            static,
            hists,
            observation,
        )

        # gradient descent step (in-place update `p.value`)
        def gd(p: evm.Parameter, g: evm.Parameter, lr: float = 1e-2) -> None:
            p.value[...] -= lr * g.value

        # apply the gradient descent step to each parameter in the dynamic part
        jax.tree.map(gd, dynamic_ref, grads, is_leaf=evmp.is_parameter)
        return loss_val

    # minimize with 5000 steps
    for step in range(5000):
        loss_val = minimize_step(dynamic_ref, static, hists, observation)
        if step % 500 == 0:
            print(f"{step=} - {loss_val=:.6f}")

    # return best fit values (immutable)
    return evmp.pure(evmm.freeze(dynamic_ref))


if __name__ == "__main__":
    bestfit_params = fit(params, hists, observation)

    print("Bestfit parameter:")
    wl.pprint(bestfit_params, short_arrays=False)
