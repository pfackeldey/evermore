import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from model import hists, model, observation, params

import evermore as evm


@eqx.filter_jit
def loss(
    params: PyTree[evm.Parameter],
    hists: PyTree[Float[Array, " nbins"]],
    observation: Float[Array, " nbins"],
) -> Float[Array, ""]:
    expectations = model(params, hists)
    constraints = evm.loss.get_log_probs(params)
    loss_val = (
        evm.pdf.PoissonContinuous(lamb=evm.util.sum_over_leaves(expectations))
        .log_prob(
            observation,
        )
        .sum()
    )
    # add constraint
    loss_val += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(loss_val)


loss_val = loss(params, hists, observation)
print(f"{loss_val=}")
grads = eqx.filter_grad(loss)(params, hists, observation)
print(
    "Gradients:",
    jax.tree.map(lambda p: p.value.item(), grads, is_leaf=evm.parameter.is_parameter),
)
