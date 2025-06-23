import equinox as eqx
import jax
import wadler_lindig as wl
from model import hists, loss, observation, params

import evermore as evm

if __name__ == "__main__":
    dynamic, static = evm.parameter.partition(params)
    loss_val = loss(dynamic, static, hists, observation)
    print(f"{loss_val=}")
    grads = eqx.filter_grad(loss)(dynamic, static, hists, observation)
    print("Gradients:")
    wl.pprint(
        jax.tree.map(lambda p: p.value, grads, is_leaf=evm.parameter.is_parameter),
        short_arrays=False,
    )
