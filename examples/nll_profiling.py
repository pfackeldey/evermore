import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jaxtyping import Array

import evermore as evm


def fixed_mu_fit(mu: Array) -> Array:
    from model import hists, model, observation

    nll = evm.loss.PoissonNLL()

    optim = optax.sgd(learning_rate=1e-2)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    model = eqx.tree_at(lambda t: t.mu.value, model, mu)

    # filter out mu from the model (no gradients will be calculated for mu!)
    # see: https://github.com/patrick-kidger/equinox/blob/main/examples/frozen_layer.ipynb
    filter_spec = jtu.tree_map(lambda _: True, model)
    filter_spec = eqx.tree_at(
        lambda tree: tree.mu.value,
        filter_spec,
        replace=False,
    )

    @eqx.filter_jit
    def loss(diff_model, static_model, hists, observation):
        model = eqx.combine(diff_model, static_model)
        expectations = model(hists)
        constraints = evm.loss.get_param_constraints(model)
        loss_val = nll(
            expectation=evm.util.sum_leaves(expectations),
            observation=observation,
        )
        # add constraint
        loss_val += evm.util.sum_leaves(constraints)
        return -2 * jnp.sum(loss_val)

    @eqx.filter_jit
    def make_step(model, opt_state, events, observation):
        # differentiate
        diff_model, static_model = eqx.partition(model, filter_spec)
        grads = eqx.filter_grad(loss)(diff_model, static_model, events, observation)
        updates, opt_state = optim.update(grads, opt_state)
        # apply nuisance parameter and DNN weight updates
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    # minimize model with 1000 steps
    for _ in range(1000):
        model, opt_state = make_step(model, opt_state, hists, observation)
    diff_model, static_model = eqx.partition(model, filter_spec)
    return loss(diff_model, static_model, hists, observation)


mus = jnp.linspace(0, 5, 11)
# for loop over mu values
for mu in mus:
    print(f"[for-loop] mu={mu:.2f} - NLL={fixed_mu_fit(jnp.array(mu)):.6f}")


# or vectorized!!!
likelihood_scan = jax.vmap(fixed_mu_fit)(mus)
for mu, nll in zip(mus, likelihood_scan, strict=False):
    print(f"[jax.vmap] mu={mu:.2f} - NLL={nll:.6f}")
