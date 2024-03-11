import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray
from model import hists, model, observation

import evermore as evm

key = jax.random.PRNGKey(0)

# generate a new model with sampled parameters according to their constraint pdfs
toymodel = evm.sample.toy_module(model, key)


# generate new expectation based on the toy model
def toy_expectation(
    key: PRNGKeyArray,
    module: eqx.Module,
    hists: dict,
) -> Array:
    toymodel = evm.sample.toy_module(model, key)
    expectations = toymodel(hists)
    return evm.util.sum_leaves(expectations)


expectation = toy_expectation(key, model, hists)


# generate a new expectations vectorized over many keys
keys = jax.random.split(key, 1000)

# vectorized toy expectation
toy_expectation_vec = jax.vmap(toy_expectation, in_axes=(0, None, None))
expectations = toy_expectation_vec(keys, model, hists)


# just sample observations with poisson
poisson_obs = evm.pdf.Poisson(observation)
sampled_observation = poisson_obs.sample(key)

# vectorized sampling
sampled_observations = jax.vmap(poisson_obs.sample)(keys)
