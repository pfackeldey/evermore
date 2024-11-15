import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from model import hists, model, observation

import evermore as evm

key = jax.random.PRNGKey(0)

# set lower and upper bounds for the mu parameter
model = eqx.tree_at(lambda t: t.mu.lower, model, jnp.array([0.0]))
model = eqx.tree_at(lambda t: t.mu.upper, model, jnp.array([10.0]))

# generate a new model with sampled parameters according to their constraint pdfs
toymodel = evm.parameter.sample(model, key)


# generate new expectation based on the toy model
def toy_expectation(
    key: PRNGKeyArray,
    module: eqx.Module,
    hists: dict,
) -> Array:
    toymodel = evm.parameter.sample(model, key)
    expectations = toymodel(hists)
    return evm.util.sum_over_leaves(expectations)


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
