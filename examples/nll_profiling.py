import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree

import evermore as evm


def fixed_mu_fit(mu: Float[Array, ""]) -> Float[Array, ""]:
    from model import hists, model, observation, params

    optim = optax.sgd(learning_rate=1e-2)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # Fix `mu` and freeze the parameter
    params = eqx.tree_at(lambda t: t.mu.value, params, mu)
    params = eqx.tree_at(lambda t: t.mu.frozen, params, True)

    @eqx.filter_jit
    def loss(
        diffable: PyTree[evm.Parameter],
        static: PyTree[evm.Parameter],
        hists: PyTree[Float[Array, " nbins"]],
        observation: Float[Array, " nbins"],
    ) -> Float[Array, ""]:
        params = evm.parameter.combine(diffable, static)
        expectations = model(params, hists)
        constraints = evm.loss.get_log_probs(params)
        loss_val = (
            evm.pdf.PoissonContinuous(lamb=evm.util.sum_over_leaves(expectations))
            .log_prob(observation)
            .sum()
        )
        # add constraint
        loss_val += evm.util.sum_over_leaves(constraints)
        return -2 * jnp.sum(loss_val)

    @eqx.filter_jit
    def make_step(
        params: PyTree[evm.Parameter],
        opt_state: PyTree,
        hists: PyTree[Float[Array, " nbins"]],
        observation: Float[Array, " nbins"],
    ) -> tuple[PyTree[evm.Parameter], PyTree]:
        diffable, static = evm.parameter.partition(params)
        grads = eqx.filter_grad(loss)(diffable, static, hists, observation)
        updates, opt_state = optim.update(grads, opt_state)
        # apply parameter updates
        diffable = eqx.apply_updates(diffable, updates)
        params = evm.parameter.combine(diffable, static)
        return params, opt_state

    # minimize params with 1000 steps
    for _ in range(1000):
        params, opt_state = make_step(params, opt_state, hists, observation)
    diffable, static = evm.parameter.partition(params)
    return loss(diffable, static, hists, observation)


mus = jnp.linspace(0, 5, 11)
# for loop over mu values
for mu in mus:
    print(f"[for-loop] mu={mu:.2f} - NLL={fixed_mu_fit(jnp.array(mu)):.6f}")


# or vectorized!!!
likelihood_scan = jax.vmap(fixed_mu_fit)(mus)
for mu, nll in zip(mus, likelihood_scan, strict=False):
    print(f"[jax.vmap] mu={mu:.2f} - NLL={nll:.6f}")
