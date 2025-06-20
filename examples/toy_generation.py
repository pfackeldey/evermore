import jax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from model import hists, model, observation, params

import evermore as evm

key = jax.random.PRNGKey(0)

# generate a new set of params according to their constraint pdfs
toy_params = evm.sample.sample_uncorrelated(params, key)


# generate new expectation based on the toy model
def toy_expectation(
    key: PRNGKeyArray,
    params: PyTree[evm.Parameter],
    hists: PyTree[Float[Array, " nbins"]],
) -> Float[Array, " nbins"]:
    toy_params = evm.sample.sample_uncorrelated(params, key)
    expectations = model(toy_params, hists)
    return evm.util.sum_over_leaves(expectations)


expectation = toy_expectation(key, params, hists)


# generate a new expectations vectorized over many keys
keys = jax.random.split(key, 1000)

# vectorized toy expectation
toy_expectation_vec = jax.vmap(toy_expectation, in_axes=(0, None, None))
expectations = toy_expectation_vec(keys, params, hists)


# just sample observations with poisson
poisson_obs = evm.pdf.PoissonDiscrete(observation)
sampled_observation = poisson_obs.sample(key)

# vectorized sampling
sampled_observations = jax.vmap(poisson_obs.sample)(keys)
