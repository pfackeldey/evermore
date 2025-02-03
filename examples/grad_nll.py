import equinox as eqx
import jax.numpy as jnp
from model import hists, model, observation

import evermore as evm


@eqx.filter_jit
def loss(model, hists, observation):
    expectations = model(hists)
    constraints = evm.loss.get_log_probs(model)
    loss_val = (
        evm.pdf.Poisson(lamb=evm.util.sum_over_leaves(expectations))
        .log_prob(
            observation,
        )
        .sum()
    )
    # add constraint
    loss_val += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(loss_val)


loss_val = loss(model, hists, observation)
grads = eqx.filter_grad(loss)(model, hists, observation)
