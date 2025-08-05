from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from model import Hist1D, hists, loss, model, observation, params

import evermore as evm

key = jax.random.PRNGKey(42)


# --- Postfit sampling ---
# use the following for correlated (postfit) sampling
# (the following creates a Covariance matrix based the number of parameter in an arbitrary pytree)


# first we have to run a fit to get the cov matrix


@eqx.filter_jit
def optx_loss(dynamic, args):
    return loss(dynamic, *args)


@eqx.filter_jit
def fit(params, hists, observation):
    solver = optx.BFGS(rtol=1e-5, atol=1e-7)

    dynamic, static = evm.tree.partition(params)

    fitresult = optx.minimise(
        optx_loss,
        solver,
        dynamic,
        has_aux=False,
        args=(static, hists, observation),
        options={},
        max_steps=10_000,
        throw=True,
    )
    return evm.tree.combine(fitresult.value, static)


# generate new expectation based on the postfit toy parameters
@eqx.filter_jit
def postfit_toy_expectation(
    key: PRNGKeyArray,
    dynamic: PyTree[evm.Parameter],
    static: PyTree[evm.Parameter],
    covariance_matrix: Float[Array, "x x"],
    n_samples: int = 1,
) -> Hist1D:
    toy_dynamic = evm.sample.sample_from_covariance_matrix(
        key=key,
        params=dynamic,
        covariance_matrix=covariance_matrix,
        n_samples=n_samples,
    )
    toy_params = evm.tree.combine(toy_dynamic, static)
    expectations = model(toy_params, hists)
    return evm.util.sum_over_leaves(expectations)


@eqx.filter_jit
def prefit_toy_expectation(params, key):
    sampled_params = evm.sample.sample_from_priors(params, key)
    expectations = model(sampled_params, hists)
    return evm.util.sum_over_leaves(expectations)


if __name__ == "__main__":
    print("Exp.:", evm.util.sum_over_leaves(model(params, hists)))
    print("Obs.:", observation)

    # --- Postfit sampling ---
    bestfit_params = fit(params, hists, observation)
    dynamic, static = evm.tree.partition(bestfit_params)

    # partial it to only depend on `params`
    loss_fn = partial(optx_loss, args=(static, hists, observation))

    fast_covariance_matrix = eqx.filter_jit(evm.loss.compute_covariance)
    covariance_matrix = fast_covariance_matrix(loss_fn, dynamic)

    # create 1 toy
    expectation = postfit_toy_expectation(key, dynamic, static, covariance_matrix)
    print("1 toy (postfit):", expectation)

    # vectorized toy expectation for 10k toys
    expectations = postfit_toy_expectation(
        key, dynamic, static, covariance_matrix, n_samples=10_000
    )
    print("Mean of 10.000 toys (postfit):", jnp.mean(expectations, axis=0))
    print("Std of 10.000 toys (postfit):", jnp.std(expectations, axis=0))

    # --- Prefit sampling ---
    # create 1 toy
    expectation = prefit_toy_expectation(params, key)
    print("1 toy (prefit):", expectation)

    # vectorized toy expectation for 10k toys
    keys = jax.random.split(key, 10_000)
    expectations = jax.vmap(prefit_toy_expectation, in_axes=(None, 0))(params, keys)
    print("Mean of 10.000 toys (prefit):", jnp.mean(expectations, axis=0))
    print("Std of 10.000 toys (prefit):", jnp.std(expectations, axis=0))

    # just sample observations with poisson
    poisson_obs = evm.pdf.PoissonDiscrete(observation)
    sampled_observation = poisson_obs.sample(key, shape=(1,))

    N = 10_000
    # vectorized sampling (standard way)
    sampled_observations = poisson_obs.sample(key, shape=(N, 1))

    # vectorized sampling (generically with `vmap`)
    keys = jax.random.split(key, N)

    def sample_obs(k):
        return poisson_obs.sample(k, shape=(1,))

    sampled_observations = jax.vmap(sample_obs)(keys)
