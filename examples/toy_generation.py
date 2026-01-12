from functools import partial

import jax
import jax.numpy as jnp
import optimistix as optx
from flax import nnx
from jaxtyping import Array, Float, PyTree
from model import Hist1D, Model, hists, loss, make_model, model, observation

import evermore as evm

rngs = nnx.Rngs(0)


# --- Postfit sampling ---
# use the following for correlated (postfit) sampling
# (the following creates a Covariance matrix based the number of parameter in an arbitrary pytree)
# first we have to run a fit to get the cov matrix


@nnx.jit
def fit(model, hists, observation):
    solver = optx.BFGS(rtol=1e-5, atol=1e-7)
    graphdef, dynamic, static = nnx.split(model, evm.filter.is_dynamic_parameter, ...)
    fitresult = optx.minimise(
        loss,
        solver,
        dynamic,
        has_aux=False,
        args=(graphdef, static, hists, observation),
        options={},
        max_steps=10_000,
        throw=True,
    )
    return nnx.merge(graphdef, fitresult.value, static)


# generate new expectation based on the postfit toy parameters
@partial(nnx.jit, static_argnames=("n_samples",))
def postfit_toy_expectation(
    rngs: nnx.Rngs,
    model: Model,
    *,
    covariance_matrix: Float[Array, "x x"],
    mask: PyTree[bool] | None = None,
    n_samples: int = 1,
) -> Hist1D:
    toy_model = evm.sample.sample_from_covariance_matrix(
        rngs=rngs,
        params=model,
        covariance_matrix=covariance_matrix,
        mask=mask,
        n_samples=n_samples,
    )

    expectations = toy_model(hists)
    return evm.util.sum_over_leaves(expectations)


@nnx.jit
def prefit_toy_expectation(rngs: nnx.Rngs, model: Model) -> Hist1D:
    sampled_model = evm.sample.sample_from_priors(rngs, model)
    expectations = sampled_model(hists)
    return evm.util.sum_over_leaves(expectations)


if __name__ == "__main__":
    model = make_model()

    print("Exp.:", evm.util.sum_over_leaves(model(hists)))
    print("Obs.:", observation)
    print()

    # --- Postfit sampling ---
    bestfit_model = fit(model, hists, observation)
    graphdef, dynamic, static = nnx.split(
        bestfit_model, evm.filter.is_dynamic_parameter, ...
    )
    args = (graphdef, static, hists, observation)

    # partial it to only depend on `dynamic`
    loss_fn = jax.tree_util.Partial(loss, args=args)

    fast_covariance_matrix = nnx.jit(evm.loss.covariance_matrix, static_argnums=(0,))
    covariance_matrix = fast_covariance_matrix(loss_fn, dynamic)

    # create 1 toy
    expectation = postfit_toy_expectation(
        rngs, model, covariance_matrix=covariance_matrix
    )
    print("1 toy (postfit):", expectation)

    # vectorized toy expectation for 10k toys
    expectations = postfit_toy_expectation(
        rngs, model, covariance_matrix=covariance_matrix, n_samples=10_000
    )
    print("Mean of 10.000 toys (postfit):", jnp.mean(expectations, axis=0))
    print("Std of 10.000 toys (postfit):", jnp.std(expectations, axis=0))
    print()

    # using a mask to only sample some parameters (here: only `norm1` and `norm2`)
    _params = nnx.state(model).filter(evm.filter.is_dynamic_parameter)
    mask = nnx.map_state(lambda _, p: p.name.startswith("norm"), _params)

    # create 1 toy
    expectation = postfit_toy_expectation(
        rngs, model, covariance_matrix=covariance_matrix, mask=mask
    )
    print("1 toy (postfit, only norm1 & norm2):", expectation)

    # vectorized toy expectation for 10k toys
    expectations = postfit_toy_expectation(
        rngs,
        model,
        covariance_matrix=covariance_matrix,
        n_samples=10_000,
        mask=mask,
    )
    print(
        "Mean of 10.000 toys (postfit, only norm1 & norm2):",
        jnp.mean(expectations, axis=0),
    )
    print(
        "Std of 10.000 toys (postfit, only norm1 & norm2):",
        jnp.std(expectations, axis=0),
    )
    print()

    # --- Prefit sampling ---
    # create 1 toy
    expectation = prefit_toy_expectation(rngs, model)
    print("1 toy (prefit):", expectation)

    # vectorized toy expectation for 10k toys
    @nnx.split_rngs(splits=10_000)
    @partial(nnx.vmap, in_axes=(0, None))
    def vectorized_prefit_toy_expectation(rngs, model):
        return prefit_toy_expectation(rngs, model)

    expectations = vectorized_prefit_toy_expectation(rngs, model)
    print("Mean of 10.000 toys (prefit):", jnp.mean(expectations, axis=0))
    print("Std of 10.000 toys (prefit):", jnp.std(expectations, axis=0))

    # just sample observations with poisson
    poisson_obs = evm.pdf.PoissonDiscrete(observation)
    sampled_observation = poisson_obs.sample(rngs(), shape=(1,))

    N = 10_000
    # vectorized sampling (standard way)
    sampled_observations = poisson_obs.sample(rngs(), shape=(N, 1))

    # vectorized sampling (generically with `vmap`)
    keys = jax.random.split(rngs(), N)

    def sample_obs(k):
        return poisson_obs.sample(k, shape=(1,))

    sampled_observations = jax.vmap(sample_obs)(keys)
