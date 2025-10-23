import jax.numpy as jnp
import optimistix as optx
from flax import nnx
from jaxtyping import Array, Float

import evermore as evm

solver = optx.BFGS(rtol=1e-5, atol=1e-7)


def fixed_mu_fit(mu: Float[Array, ""]) -> Float[Array, ""]:
    from model import hists, loss, observation, params

    # create a new params instance in each fit to avoid side effects
    params = params()

    # Freeze the parameter and update the `mu` value in the params tree:
    params.mu.value = mu
    params.mu.set_metadata(frozen=True)

    graphdef, dynamic, static = nnx.split(params, evm.filter.is_dynamic_parameter, ...)

    def twice_nll(dynamic, args):
        graphdef, static, hists, observation = args
        params = nnx.merge(graphdef, dynamic, static)
        return 2 * loss(params, hists=hists, observation=observation)

    fitresult = optx.minimise(
        twice_nll,
        solver,
        dynamic,
        has_aux=False,
        args=(graphdef, static, hists, observation),
        options={},
        max_steps=10_000,
        throw=True,
    )

    return twice_nll(fitresult.value, (graphdef, static, hists, observation))


if __name__ == "__main__":
    mus = jnp.linspace(0, 5, 11)
    # for loop over mu values
    for mu in mus:
        print(f"[for-loop] mu={mu:.2f} - NLL={fixed_mu_fit(jnp.array(mu)):.6f}")

    # or vectorized!!!
    likelihood_scan = nnx.vmap(fixed_mu_fit)(mus)
    for mu, nll in zip(mus, likelihood_scan, strict=False):
        print(f"[jax.vmap] mu={mu:.2f} - NLL={nll:.6f}")
