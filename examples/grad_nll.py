import wadler_lindig as wl
from flax import nnx
from model import hists, observation, params
from model import loss as _loss

import evermore as evm


def loss(dynamic, args):
    (graphdef, static, hists, observation) = args
    params = nnx.merge(graphdef, dynamic, static)
    return _loss(params, hists=hists, observation=observation)


if __name__ == "__main__":
    graphdef, dynamic, static = nnx.split(params, evm.filter.is_dynamic_parameter, ...)
    args = (graphdef, static, hists, observation)
    loss_val = loss(dynamic, args)
    print(f"{loss_val=}")
    grads = nnx.grad(loss)(dynamic, args)
    print("Gradients:")
    wl.pprint(
        nnx.pure(grads),
        short_arrays=False,
    )
