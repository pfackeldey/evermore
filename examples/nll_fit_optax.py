import optax
import wadler_lindig as wl
from flax import nnx
from model import hists, loss, make_model, observation

import evermore as evm


def fit(model, hists, observation):
    # split the model into metadata (graphdef), dynamic and static part
    graphdef, dynamic, static = nnx.split(model, evm.filter.is_dynamic_parameter, ...)
    args = (graphdef, static, hists, observation)

    tx = optax.sgd(learning_rate=1e-2)
    optimizer = nnx.Optimizer(dynamic, tx, wrt=evm.filter.is_dynamic_parameter)

    @nnx.jit
    def make_step(optimizer, dynamic):
        grads = nnx.grad(loss)(dynamic, args)
        optimizer.update(dynamic, grads)
        return dynamic

    # minimize params with 5000 steps
    for step in range(5000):
        if step % 500 == 0:
            loss_val = loss(dynamic, args)
            print(f"{step=} - {loss_val=:.6f}")

        dynamic = make_step(optimizer, dynamic)

    # combine optimized dynamic part with the static pytree
    return nnx.merge(graphdef, dynamic, static)


if __name__ == "__main__":
    model = make_model()
    bestfit_params = fit(model, hists, observation)

    print("Bestfit parameter:")
    wl.pprint(nnx.pure(bestfit_params), short_arrays=False)
