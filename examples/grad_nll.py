import equinox as eqx
import wadler_lindig as wl
from model import hists, loss, observation, params

import evermore as evm

if __name__ == "__main__":
    dynamic, static = evm.tree.partition(params)
    loss_val = loss(dynamic, static, hists, observation)
    print(f"{loss_val=}")
    grads = eqx.filter_grad(loss)(dynamic, static, hists, observation)
    print("Gradients:")
    wl.pprint(
        evm.tree.pure(grads),
        short_arrays=False,
    )
