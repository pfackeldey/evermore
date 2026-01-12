from functools import partial

import wadler_lindig as wl
from flax import nnx
from model import hists, loss, make_model, observation

import evermore as evm

if __name__ == "__main__":
    model = make_model()
    # split the model into metadata (graphdef), dynamic and static part
    graphdef, dynamic, static = nnx.split(model, evm.filter.is_dynamic_parameter, ...)
    # create a loss function depending only on the dynamic part
    args = (graphdef, static, hists, observation)
    loss_fn = partial(loss, args=args)
    # eval and differentiate loss_fn
    loss_val = loss_fn(dynamic)
    print(f"{loss_val=}")
    grads = nnx.grad(loss_fn)(dynamic)
    print("Gradients:")
    wl.pprint(
        nnx.pure(grads),
        short_arrays=False,
    )
