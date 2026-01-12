import jax.numpy as jnp
import optimistix as optx
from flax import nnx
from jaxtyping import Array, Float
from model import hists, loss, make_model, observation

import evermore as evm

solver = optx.BFGS(rtol=1e-5, atol=1e-7)


@nnx.jit
def fixed_mu_fit(mu: Float[Array, ""]) -> Float[Array, ""]:
    model = make_model()

    # Set & freeze the `mu` parameter in the model (it's mutable!):
    model.mu[...] = mu
    model.mu.set_metadata(frozen=True)

    # split the model into metadata (graphdef), dynamic and static part
    graphdef, dynamic, static = nnx.split(model, evm.filter.is_dynamic_parameter, ...)

    def twice_nll(dynamic, args):
        return 2 * loss(dynamic, args)

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
    # for-loop over mu values
    for mu in mus:
        nll = fixed_mu_fit(mu)
        print(f"[for-loop] mu={mu:.2f} - NLL={nll:.6f}")

    print("---------------------------------")

    # or vectorized!!!
    likelihood_scan = nnx.vmap(fixed_mu_fit, in_axes=(0,))(mus)
    for mu, nll in zip(mus, likelihood_scan, strict=False):
        print(f"[jax.vmap] mu={mu:.2f} - NLL={nll:.6f}")
