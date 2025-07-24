import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float

import evermore as evm

solver = optx.BFGS(rtol=1e-5, atol=1e-7)


def fixed_mu_fit(mu: Float[Array, ""]) -> Float[Array, ""]:
    from model import hists, loss, observation, params

    # Fix `mu` and freeze the parameter
    params = eqx.tree_at(lambda t: t.mu.value, params, mu)
    params = eqx.tree_at(lambda t: t.mu.frozen, params, True)

    dynamic, static = evm.parameter.partition(params)

    def twice_nll(dynamic, args):
        return 2.0 * loss(dynamic, *args)

    fitresult = optx.minimise(
        twice_nll,
        solver,
        dynamic,
        has_aux=False,
        args=(static, hists, observation),
        options={},
        max_steps=10_000,
        throw=True,
    )

    return twice_nll(fitresult.value, (static, hists, observation))


if __name__ == "__main__":
    mus = jnp.linspace(0, 5, 11)
    # for loop over mu values
    for mu in mus:
        print(f"[for-loop] mu={mu:.2f} - NLL={fixed_mu_fit(jnp.array(mu)):.6f}")

    # or vectorized!!!
    likelihood_scan = jax.vmap(fixed_mu_fit)(mus)
    for mu, nll in zip(mus, likelihood_scan, strict=False):
        print(f"[jax.vmap] mu={mu:.2f} - NLL={nll:.6f}")
