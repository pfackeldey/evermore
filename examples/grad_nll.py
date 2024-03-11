import equinox as eqx
import jax.numpy as jnp
from model import hists, model, observation

import evermore as evm

nll = evm.loss.PoissonNLL()


@eqx.filter_jit
def loss(model, hists, observation):
    expectations = model(hists)
    constraints = evm.loss.get_param_constraints(model)
    loss_val = nll(
        expectation=evm.util.sum_leaves(expectations),
        observation=observation,
    )
    # add constraint
    loss_val += evm.util.sum_leaves(constraints)
    return -jnp.sum(loss_val)


loss_val = loss(model, hists, observation)
grads = eqx.filter_grad(loss)(model, hists, observation)
