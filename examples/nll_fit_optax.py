import optax
import wadler_lindig as wl
from flax import nnx
from model import hists, loss, observation, params

import evermore as evm


def fit(params, hists, observation):
    graphdef, dynamic, static = nnx.split(params, evm.filter.is_dynamic_parameter, ...)

    tx = optax.sgd(learning_rate=1e-2)
    optimizer = nnx.Optimizer(dynamic, tx, wrt=evm.filter.is_dynamic_parameter)

    def tx_loss(dynamic):
        params = nnx.merge(graphdef, dynamic, static, copy=True)
        return loss(
            params,
            hists=hists,
            observation=observation,
        )

    @nnx.jit
    def make_step(optimizer, dynamic):
        grads = nnx.grad(tx_loss)(dynamic)
        optimizer.update(dynamic, grads)
        return dynamic

    # minimize params with 5000 steps
    for step in range(5000):
        if step % 500 == 0:
            loss_val = tx_loss(dynamic)
            print(f"{step=} - {loss_val=:.6f}")

        dynamic = make_step(optimizer, dynamic)

    # combine optimized dynamic part with the static pytree
    return nnx.merge(graphdef, dynamic, static)


if __name__ == "__main__":
    bestfit_params = fit(params, hists, observation)

    print("Bestfit parameter:")
    wl.pprint(nnx.pure(bestfit_params), short_arrays=False)
