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
        dynamic: PyTree[evm.Parameter],
        static: PyTree[evm.Parameter],
        hists: PyTree[Float[Array, " nbins"]],
        observation: Float[Array, " nbins"],
    ) -> Float[Array, ""]:
        params = evm.parameter.combine(dynamic, static)
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
        dynamic: PyTree[evm.Parameter],
        static: PyTree[evm.Parameter],
        opt_state: PyTree,
        hists: PyTree[Float[Array, " nbins"]],
        observation: Float[Array, " nbins"],
    ) -> tuple[PyTree[evm.Parameter], PyTree]:
        grads = eqx.filter_grad(loss)(dynamic, static, hists, observation)
        updates, opt_state = optim.update(grads, opt_state)
        # apply parameter updates
        dynamic = eqx.apply_updates(dynamic, updates)
        return dynamic, opt_state

    dynamic, static = evm.parameter.partition(params)

    # minimize params with 1000 steps
    for _ in range(1000):
        dynamic, opt_state = make_step(dynamic, static, opt_state, hists, observation)
    return loss(dynamic, static, hists, observation)


if __name__ == "__main__":
    mus = jnp.linspace(0, 5, 11)
    # for loop over mu values
    for mu in mus:
        print(f"[for-loop] mu={mu:.2f} - NLL={fixed_mu_fit(jnp.array(mu)):.6f}")

    # or vectorized!!!
    likelihood_scan = jax.vmap(fixed_mu_fit)(mus)
    for mu, nll in zip(mus, likelihood_scan, strict=False):
        print(f"[jax.vmap] mu={mu:.2f} - NLL={nll:.6f}")
