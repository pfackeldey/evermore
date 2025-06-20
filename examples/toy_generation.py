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
    # differentiate full analysis
    diffable, static = evm.parameter.partition(params)
    grads = eqx.filter_grad(loss)(diffable, static, hists, observation)
    updates, opt_state = optim.update(grads, opt_state)
    # apply nuisance parameter and DNN weight updates
    params = eqx.apply_updates(params, updates)
    return params, opt_state


# minimize params with 3.000 steps
for step in range(3_000):
    if step % 500 == 0:
        diffable, static = evm.parameter.partition(params)
        loss_val = loss(diffable, static, hists, observation)
        print(f"{step=} - {loss_val=:.6f}")
    params, opt_state = make_step(params, opt_state, hists, observation)

diffable, static = evm.parameter.partition(params)
fast_covariance_matrix = eqx.filter_jit(evm.sample.compute_covariance_matrix)
covariance_matrix = fast_covariance_matrix(
    loss=loss,
    params=diffable,
    args=(static, hists, observation),
)


# --- Prefit sampling ---
# use the following instead of the code above for decorrelated (prefit) sampling:
# (the following creates a Identity matrix based the number of parameter in an arbitrary pytree)
# values = jax.tree.map(lambda p: p.value, params, is_leaf=evm.parameter.is_parameter)
# flat_values, _ = jax.flatten_util.ravel_pytree(values)
# covariance_matrix = jnp.eye(flat_values.shape[0])


# generate new expectation based on the toy parameters
# @eqx.filter_jit
def toy_expectation(
    key: PRNGKeyArray,
    params: PyTree[evm.Parameter],
    *,
    n_samples: int = 1,
) -> Float[Array, " nbins"]:
    toy_params = evm.sample.sample_from_covariance_matrix(
        key=key,
        params=params,
        covariance_matrix=covariance_matrix,
        n_samples=n_samples,
    )
    expectations = model(toy_params, hists)
    return evm.util.sum_over_leaves(expectations)


print("Exp.:", evm.util.sum_over_leaves(model(params, hists)))
print("Obs.:", observation)

# create 1 toy
expectation = toy_expectation(key, params)
print("1 toy:", expectation)

# vectorized toy expectation for 10k toys
expectations = toy_expectation(key, params, n_samples=10_000)
print("Mean of 10.000 toys:", jnp.mean(expectations, axis=0))
print("Std of 10.000 toys:", jnp.std(expectations, axis=0))


# just sample observations with poisson
poisson_obs = evm.pdf.PoissonDiscrete(observation)
sampled_observation = poisson_obs.sample(key)

# vectorized sampling (generically with `vmap`)
keys = jax.random.split(key, 10_000)
sampled_observations = jax.vmap(poisson_obs.sample)(keys)
