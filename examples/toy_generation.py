import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from model import hists, model, observation, params

import evermore as evm

key = jax.random.PRNGKey(42)


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
        evm.pdf.PoissonContinuous(evm.util.sum_over_leaves(expectations))
        .log_prob(observation)
        .sum()
    )
    # add constraint
    loss_val += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(loss_val)


# --- Postfit sampling ---
# use the following for correlated (postfit) sampling
# (the following creates a Covariance matrix based the number of parameter in an arbitrary pytree)

# First we have to optimize of course out parameters, let's copy & past from examples/nll_fit.py
import optax  # noqa: E402

optim = optax.sgd(learning_rate=1e-2)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


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


# minimize params with 3.000 steps
for step in range(3_000):
    if step % 500 == 0:
        diffable, static = evm.parameter.partition(params)
        loss_val = loss(diffable, static, hists, observation)
        print(f"{step=} - {loss_val=:.6f}")
    params, opt_state = make_step(params, opt_state, hists, observation)

diffable, static = evm.parameter.partition(params)
fast_covariance_matrix = eqx.filter_jit(evm.loss.compute_covariance)


# partial it to only depend on `params`
def loss_fn(params):
    return loss(params, static, hists, observation)


covariance_matrix = fast_covariance_matrix(loss_fn, diffable)


# generate new expectation based on the postfit toy parameters
@eqx.filter_jit
def postfit_toy_expectation(
    key: PRNGKeyArray,
    diffable: PyTree[evm.Parameter],
    static: PyTree[evm.Parameter],
    *,
    n_samples: int = 1,
) -> Float[Array, " nbins"]:
    toy_diffable = evm.sample.sample_from_covariance_matrix(
        key=key,
        params=diffable,
        covariance_matrix=covariance_matrix,
        n_samples=n_samples,
    )
    toy_params = evm.parameter.combine(toy_diffable, static)
    expectations = model(toy_params, hists)
    return evm.util.sum_over_leaves(expectations)


print("Exp.:", evm.util.sum_over_leaves(model(params, hists)))
print("Obs.:", observation)

# create 1 toy
expectation = postfit_toy_expectation(key, diffable, static)
print("1 toy (postfit):", expectation)

# vectorized toy expectation for 10k toys
expectations = postfit_toy_expectation(key, diffable, static, n_samples=10_000)
print("Mean of 10.000 toys (postfit):", jnp.mean(expectations, axis=0))
print("Std of 10.000 toys (postfit):", jnp.std(expectations, axis=0))


# --- Prefit sampling ---
# use the following code for decorrelated (prefit) sampling:
# (samples from each parameters prior pdf)


def prefit_toy_expectation(params, key):
    sampled_params = evm.sample.sample_from_priors(params, key)
    expectations = model(sampled_params, hists)
    return evm.util.sum_over_leaves(expectations)


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
sampled_observation = poisson_obs.sample(key)

# vectorized sampling (generically with `vmap`)
keys = jax.random.split(key, 10_000)
sampled_observations = jax.vmap(poisson_obs.sample)(keys)
