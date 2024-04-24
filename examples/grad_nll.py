import equinox as eqx
import jax.numpy as jnp
from model import hists, model, observation

import evermore as evm

log_likelihood = evm.loss.PoissonLogLikelihood()


@eqx.filter_jit
def loss(model, hists, observation):
    expectations = model(hists)
    constraints = evm.loss.get_log_probs(model)
    loss_val = log_likelihood(
        expectation=evm.util.sum_over_leaves(expectations),
        observation=observation,
    )
    # add constraint
    loss_val += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(loss_val)


loss_val = loss(model, hists, observation)
grads = eqx.filter_grad(loss)(model, hists, observation)
